from pathlib import Path

import pytest

from ragu.adapters.ingestion import OcrLine, OcrPage, OcrWord, dump_documents, load_paths


class FakeOcr:
    name = "fake-ocr"

    def read_page(self, image: bytes, *, index: int = 0) -> OcrPage:
        words = [
            OcrWord(text="scanned", box=(0, 0, 50, 10)),
            OcrWord(text="invoice", box=(52, 0, 100, 10)),
            OcrWord(text="42", box=(102, 0, 120, 10)),
        ]
        line = OcrLine(
            text="scanned invoice 42", score=0.99, poly=[(0, 0), (120, 0), (120, 10), (0, 10)],
            box=(0, 0, 120, 10), words=words,
        )
        return OcrPage(index=index, source="ocr", text="scanned invoice 42",
                       width=200, height=20, lines=[line])


def test_load_text_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Title\n\nHello.")
    (tmp_path / "b.txt").write_text("plain text")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")

    docs = load_paths([tmp_path])

    ids = sorted(str(d.id) for d in docs)
    assert ids == ["a.md", "b.txt"]  # .bin skipped
    assert all(d.metadata["loader"] == "text" for d in docs)


def test_ocr_dispatch_with_engine(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("typed note")
    (tmp_path / "scan.png").write_bytes(b"\x89PNG fake bytes")

    without = load_paths([tmp_path])
    assert sorted(str(d.id) for d in without) == ["note.txt"]  # png skipped (no engine)

    with_ocr = load_paths([tmp_path], ocr=FakeOcr())
    by_id = {str(d.id): d for d in with_ocr}
    assert set(by_id) == {"note.txt", "scan.png"}
    scan = by_id["scan.png"]
    assert scan.content == "scanned invoice 42"
    assert scan.metadata["loader"] == "ocr-image"
    # Full structured OCR is preserved on the document.
    ocr = scan.artifacts["ocr"]
    assert ocr["engine"] == "fake-ocr"
    words = ocr["pages"][0]["lines"][0]["words"]
    assert [w["text"] for w in words] == ["scanned", "invoice", "42"]
    assert tuple(words[0]["box"]) == (0, 0, 50, 10)  # boxes survive serialization


def test_text_layer_pdf_has_word_geometry(tmp_path: Path) -> None:
    """A born-digital PDF (no OCR) yields the same structured shape as OCR:
    lines with word boxes, scaled to the DPI pixel space, source='text-layer'."""
    fitz = pytest.importorskip("fitz")
    from ragu.adapters.ingestion.pdf import load_pdf_file

    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world layer", fontsize=14)
    doc.save(pdf)
    doc.close()

    # mode='text' guarantees the text-layer path even with an engine present.
    document = load_pdf_file(pdf, tmp_path, ocr=None, mode="text", dpi=200)
    artifact = document.artifacts["ocr"]
    assert document.metadata["loader"] == "pdf-text"

    page0 = artifact["pages"][0]
    assert page0["source"] == "text-layer"
    assert page0["lines"], "text-layer page should carry line geometry"
    line = page0["lines"][0]
    assert line["text"] == "Hello world layer"
    assert line["score"] == 1.0  # text layer is exact, not recognized
    words = [w["text"] for w in line["words"]]
    assert words == ["Hello", "world", "layer"]
    # Boxes are 4-int axis-aligned and within the (DPI-scaled) page bounds.
    x0, y0, x1, y1 = line["box"]
    assert x0 < x1 and y0 < y1
    assert x1 <= page0["width"] and y1 <= page0["height"]


def test_same_name_different_folders_get_distinct_ids(tmp_path: Path) -> None:
    """Same filename in different folders must never collide on one id, however
    the index command is invoked (parent dir, sibling dirs, or files directly)."""
    from ragu.adapters.ingestion import iter_files

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "doc.txt").write_text("alpha")
    (tmp_path / "b" / "doc.txt").write_text("bravo")

    def ids(paths: list[Path]) -> list[str]:
        return sorted(f.relative_to(base).as_posix() for f, base in iter_files(paths))

    expected = ["a/doc.txt", "b/doc.txt"]
    assert ids([tmp_path]) == expected  # index the parent
    assert ids([tmp_path / "a", tmp_path / "b"]) == expected  # sibling dirs
    assert ids(
        [tmp_path / "a" / "doc.txt", tmp_path / "b" / "doc.txt"]
    ) == expected  # files directly
    # Overlapping arguments ingest each file once.
    assert ids([tmp_path, tmp_path / "a" / "doc.txt"]) == expected


def test_recursive_relpath_and_extract(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sub" / "deep").mkdir(parents=True)
    (src / "top.md").write_text("top doc")
    (src / "sub" / "nested.txt").write_text("nested doc")
    (src / "sub" / "deep" / "scan.png").write_bytes(b"fake")

    docs = load_paths([src], ocr=FakeOcr())
    relpaths = {d.metadata["rel_path"] for d in docs}
    assert relpaths == {"top.md", "sub/nested.txt", "sub/deep/scan.png"}

    out = tmp_path / "out"
    written = dump_documents(docs, out)
    assert written == 3
    # Structure is mirrored, with .txt appended.
    assert (out / "top.md.txt").read_text() == "top doc"
    assert (out / "sub" / "nested.txt.txt").read_text() == "nested doc"
    assert (out / "sub" / "deep" / "scan.png.txt").read_text() == "scanned invoice 42"
    # OCR'd files also get a sidecar JSON with the full geometry; text files don't.
    assert (out / "sub" / "deep" / "scan.png.ocr.json").exists()
    assert not (out / "top.md.ocr.json").exists()

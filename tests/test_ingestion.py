from pathlib import Path

from ragu.adapters.ingestion import dump_documents, load_paths


class FakeOcr:
    def read_image_bytes(self, image: bytes) -> str:
        return "scanned invoice total 42"


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
    assert by_id["scan.png"].content == "scanned invoice total 42"
    assert by_id["scan.png"].metadata["loader"] == "ocr-image"


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
    assert (out / "sub" / "deep" / "scan.png.txt").read_text() == "scanned invoice total 42"

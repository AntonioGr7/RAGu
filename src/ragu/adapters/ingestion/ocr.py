"""OCR ingestion via PP-OCRv6 (PaddleOCR), behind a swappable ``OcrEngine``.

PP-OCRv6 (https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) ships a
detection ("det") and a recognition ("rec") model in three tiers
(tiny/small/medium). Detection finds text regions; recognition reads each
region. We expose both as configuration and pass them to PaddleOCR as model
directories, so any tier (or a future model) works without code changes.

The concrete engine and its heavy deps (paddleocr, pymupdf) are imported lazily
and live behind the optional ``ocr`` extra. Everything here depends only on the
``OcrEngine`` Protocol, so tests substitute a trivial fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ragu.adapters.ingestion.files import make_document
from ragu.core import Document

# Raster image formats handled directly by OCR. PDFs are rasterized page-by-page.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_SUFFIXES = {".pdf"}
OCR_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES


@runtime_checkable
class OcrEngine(Protocol):
    """Reads text from a raster image given as raw bytes."""

    def read_image_bytes(self, image: bytes) -> str:
        ...


class PaddleOcrEngine:
    """PP-OCRv6 detection + recognition via the PaddleOCR 3.x pipeline.

    Tier (tiny/small/medium) is selected by model *name*
    (e.g. ``PP-OCRv6_small_det`` / ``PP-OCRv6_small_rec``); PaddleOCR downloads
    them on first use. ``*_model_dir`` overrides with a local directory. Leaving
    both None uses PaddleOCR's defaults, which are already PP-OCRv6.

    ``enable_mkldnn=False`` is the default here: paddlepaddle 3.x's oneDNN CPU
    path crashes on the PP-OCRv6 detection graph, and disabling it is the
    supported workaround.
    """

    def __init__(
        self,
        *,
        det_model_name: str | None = None,
        rec_model_name: str | None = None,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        lang: str = "en",
        device: str = "cpu",
        enable_mkldnn: bool = False,
        det_limit_side_len: int | None = None,
        rec_batch_size: int | None = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PaddleOcrEngine requires the 'ocr' extra: pip install 'ragu[ocr]'"
            ) from exc
        # Only pass tuning knobs when set, so PaddleOCR's own defaults apply.
        extra: dict[str, object] = {}
        if det_limit_side_len is not None:
            extra["text_det_limit_side_len"] = det_limit_side_len
        if rec_batch_size is not None:
            extra["text_recognition_batch_size"] = rec_batch_size
        self._ocr = PaddleOCR(
            text_detection_model_name=det_model_name,
            text_detection_model_dir=det_model_dir,
            text_recognition_model_name=rec_model_name,
            text_recognition_model_dir=rec_model_dir,
            lang=lang,
            device=device,
            enable_mkldnn=enable_mkldnn,
            # Page-level preprocessing stages we don't need for plain text recall.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **extra,
        )

    def read_image_bytes(self, image: bytes) -> str:
        import io

        import numpy as np
        from PIL import Image

        arr = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        return _join_lines(self._ocr.predict(arr))


def _join_lines(results: object) -> str:
    """Flatten PaddleOCR 3.x output to text lines.

    ``predict`` returns one result mapping per input image; each carries a
    ``rec_texts`` list of recognized lines in reading order.
    """
    lines: list[str] = []
    for page in results or []:  # type: ignore[union-attr]
        texts = page.get("rec_texts") if hasattr(page, "get") else None
        for text in texts or []:
            if text:
                lines.append(str(text))
    return "\n".join(lines)


def load_ocr_file(file: Path, base: Path, engine: OcrEngine) -> Document:
    """OCR a single image or PDF file into a Document."""
    suffix = file.suffix.lower()
    if suffix in PDF_SUFFIXES:
        pages = [engine.read_image_bytes(img) for img in _pdf_to_images(file)]
        content = "\n\n".join(f"[page {i + 1}]\n{p}" for i, p in enumerate(pages))
        loader = "ocr-pdf"
    else:
        content = engine.read_image_bytes(file.read_bytes())
        loader = "ocr-image"
    return make_document(file, base, content, {"loader": loader})


def _pdf_to_images(file: Path, dpi: int = 200) -> list[bytes]:
    """Rasterize each PDF page to PNG bytes via PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDF OCR requires the 'ocr' extra: pip install 'ragu[ocr]'") from exc
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[bytes] = []
    with fitz.open(file) as doc:
        for page in doc:
            images.append(page.get_pixmap(matrix=matrix).tobytes("png"))
    return images

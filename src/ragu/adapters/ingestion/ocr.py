"""OCR ingestion via PP-OCRv6 (PaddleOCR), behind a swappable ``OcrEngine``.

PP-OCRv6 (https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) ships a
detection ("det") and a recognition ("rec") model in three tiers
(tiny/small/medium). Detection finds text regions; recognition reads each
region. Tier is selected by model name; the models download on first use.

The engine returns the *full* structured result — per line: the detection
polygon, axis-aligned box, recognition confidence, and (with word boxes on) the
per-word text and boxes — not just flat text. Nothing PaddleOCR produces is
discarded; downstream can keep it as document artifacts or persist it elsewhere.

Heavy deps (paddleocr, pymupdf) are imported lazily and live behind the optional
``ocr`` extra. Everything here depends only on the ``OcrEngine`` Protocol, so
tests substitute a trivial fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ragu.adapters.ingestion.files import make_document
from ragu.core import Document

# Raster image formats handled directly by OCR. PDFs are rasterized page-by-page.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_SUFFIXES = {".pdf"}
OCR_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES

# Box = axis-aligned (x1, y1, x2, y2); Point = (x, y); Poly = 4 points.
Box = tuple[int, int, int, int]
Point = tuple[int, int]


class OcrWord(BaseModel):
    """A single recognized word with its axis-aligned box."""

    model_config = {"frozen": True}

    text: str
    box: Box


class OcrLine(BaseModel):
    """A recognized text line with detection geometry, score, and words."""

    model_config = {"frozen": True}

    text: str
    score: float
    poly: list[Point]  # 4-point detection polygon (may be rotated)
    box: Box  # axis-aligned bounding box
    words: list[OcrWord] = Field(default_factory=list)


class OcrPage(BaseModel):
    """One page/image of OCR output (or a passthrough text-layer page)."""

    model_config = {"frozen": True}

    index: int
    source: str  # "ocr" | "text-layer"
    text: str
    width: int | None = None
    height: int | None = None
    lines: list[OcrLine] = Field(default_factory=list)


class OcrArtifact(BaseModel):
    """The full structured OCR for a document — stored under
    ``Document.artifacts['ocr']`` and dumped as a sidecar by ``extract``."""

    engine: str
    pages: list[OcrPage]


@runtime_checkable
class OcrEngine(Protocol):
    """Recognizes text from a raster image, returning full structured output."""

    name: str

    def read_page(self, image: bytes, *, index: int = 0) -> OcrPage:
        ...


class PaddleOcrEngine:
    """PP-OCRv6 detection + recognition via the PaddleOCR 3.x pipeline.

    Tier (tiny/small/medium) is selected by model *name*
    (e.g. ``PP-OCRv6_small_det`` / ``PP-OCRv6_small_rec``). ``*_model_dir``
    overrides with a local directory. None uses PaddleOCR's defaults (PP-OCRv6).

    ``enable_mkldnn=False`` is the default: paddlepaddle 3.x's oneDNN CPU path
    crashes on the PP-OCRv6 detection graph, and disabling it is the supported
    workaround (ignored on GPU). ``return_word_box`` is on so word-level boxes
    are always available.
    """

    name = "paddleocr/PP-OCRv6"

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
        det_limit_type: str | None = None,
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
        if det_limit_type is not None:
            extra["text_det_limit_type"] = det_limit_type
        if rec_batch_size is not None:
            extra["text_recognition_batch_size"] = rec_batch_size
        # `lang` is ignored (and warns) once explicit model names are given, so
        # only pass it when relying on PaddleOCR's language-default models.
        if det_model_name is None and rec_model_name is None:
            extra["lang"] = lang
        self._ocr = PaddleOCR(
            text_detection_model_name=det_model_name,
            text_detection_model_dir=det_model_dir,
            text_recognition_model_name=rec_model_name,
            text_recognition_model_dir=rec_model_dir,
            device=_paddle_device(device),
            enable_mkldnn=enable_mkldnn,
            return_word_box=True,
            # Page-level preprocessing stages we don't need for plain text recall.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **extra,
        )

    def read_page(self, image: bytes, *, index: int = 0) -> OcrPage:
        import io

        import numpy as np
        from PIL import Image

        pil = Image.open(io.BytesIO(image)).convert("RGB")
        arr = np.array(pil)
        results = self._ocr.predict(arr)
        result = results[0] if results else {}
        return _page_from_result(result, index=index, width=pil.width, height=pil.height)


def _paddle_device(device: str) -> str:
    """Map a torch-style device to PaddlePaddle's naming.

    PaddlePaddle uses "gpu"/"gpu:0", not torch's "cuda"/"cuda:0". Accepting
    "cuda" here means one device name (RAGU_*__DEVICE) works for both the
    embedder (torch) and OCR (paddle)."""
    d = device.strip().lower()
    if d == "cuda":
        return "gpu"
    if d.startswith("cuda:"):
        return "gpu:" + d.split(":", 1)[1]
    return device


def _page_from_result(result: object, *, index: int, width: int, height: int) -> OcrPage:
    """Build an OcrPage from a PaddleOCR 3.x OCRResult mapping."""
    get = result.get if hasattr(result, "get") else (lambda _k, _d=None: _d)  # type: ignore[union-attr]
    texts = get("rec_texts") or []
    scores = get("rec_scores") or []
    polys = get("rec_polys") or []
    boxes = get("rec_boxes")
    word_tokens = get("text_word") or []
    word_boxes = get("text_word_boxes") or []

    lines: list[OcrLine] = []
    for i, text in enumerate(texts):
        words: list[OcrWord] = []
        if i < len(word_tokens) and i < len(word_boxes):
            for tok, wb in zip(word_tokens[i], word_boxes[i], strict=False):
                if str(tok).strip():
                    words.append(OcrWord(text=str(tok), box=_to_box(wb)))
        lines.append(
            OcrLine(
                text=str(text),
                score=float(scores[i]) if i < len(scores) else 0.0,
                poly=_to_poly(polys[i]) if i < len(polys) else [],
                box=_to_box(boxes[i]) if boxes is not None and i < len(boxes) else (0, 0, 0, 0),
                words=words,
            )
        )

    return OcrPage(
        index=index,
        source="ocr",
        text="\n".join(line.text for line in lines if line.text),
        width=width,
        height=height,
        lines=lines,
    )


def _to_box(raw: object) -> Box:
    vals = [int(v) for v in list(raw)]  # type: ignore[call-overload]
    return (vals[0], vals[1], vals[2], vals[3])


def _to_poly(raw: object) -> list[Point]:
    return [(int(p[0]), int(p[1])) for p in list(raw)]  # type: ignore[index]


def load_ocr_file(file: Path, base: Path, engine: OcrEngine) -> Document:
    """OCR a single image into a Document, attaching the structured result."""
    page = engine.read_page(file.read_bytes(), index=0)
    artifact = OcrArtifact(engine=engine.name, pages=[page])
    return make_document(
        file,
        base,
        content=page.text,
        extra={"loader": "ocr-image"},
        artifacts={"ocr": artifact.model_dump()},
    )

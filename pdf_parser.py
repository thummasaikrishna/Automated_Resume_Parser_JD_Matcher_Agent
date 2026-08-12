"""
PDF / image resume parsing.

Primary text extract: PyMuPDF, fallback pypdf.
Image-only / scanned resumes: RapidOCR (page render + embedded images).
Also accepts PNG/JPG uploads.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class ParsedResume:
    text: str
    page_count: int
    char_count: int
    filename: str
    parser: str = "unknown"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_ocr_engine() -> Any:
    """Create RapidOCR once — repeated init is slow and flaky under Streamlit."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_pil_image(img: Any) -> str:
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError("Expected a PIL Image")

    img = img.convert("RGB")
    if max(img.size) < 1400:
        scale = 1400 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    engine = _get_ocr_engine()
    result, _ = engine(buf.getvalue())
    if not result:
        return ""
    lines = [row[1] for row in result if row and len(row) > 1 and row[1]]
    return _normalize_whitespace("\n".join(lines))


def _ocr_image_bytes(image_bytes: bytes) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    return _ocr_pil_image(img)


def _ocr_pdf_pages(file_bytes: bytes) -> tuple[str, int]:
    """
    OCR every page of an image-only PDF.

    Tries embedded images first (common for screenshot resumes), then
    falls back to rendering each page to a bitmap.
    """
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        parts: list[str] = []
        for page in doc:
            page_text_bits: list[str] = []

            # 1) Embedded images (screenshot-style PDFs)
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    extracted = doc.extract_image(xref)
                    img_bytes = extracted.get("image") or b""
                    if img_bytes:
                        t = _ocr_image_bytes(img_bytes)
                        if t:
                            page_text_bits.append(t)
                except Exception:
                    continue

            # 2) Page rasterization if embedded images were missing/weak
            combined = _normalize_whitespace("\n".join(page_text_bits))
            if len(combined) < 40:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                t = _ocr_pil_image(img)
                if t:
                    page_text_bits.append(t)

            parts.append(_normalize_whitespace("\n".join(page_text_bits)))

        text = _normalize_whitespace("\n\n".join(parts))
        return text, doc.page_count
    finally:
        doc.close()


def _parse_with_pymupdf(file_bytes: bytes, filename: str) -> ParsedResume:
    import pymupdf

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        pages = [page.get_text("text") or "" for page in doc]
        text = _normalize_whitespace("\n".join(pages))
        page_count = doc.page_count
    finally:
        doc.close()

    if len(text) < 40:
        raise ValueError("PyMuPDF extracted too little selectable text (likely an image-only PDF).")

    return ParsedResume(
        text=text,
        page_count=page_count,
        char_count=len(text),
        filename=filename or "resume.pdf",
        parser="pymupdf",
    )


def _parse_with_pypdf(file_bytes: bytes, filename: str) -> ParsedResume:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = _normalize_whitespace("\n".join(pages))
    if len(text) < 40:
        raise ValueError("pypdf extracted too little selectable text.")

    return ParsedResume(
        text=text,
        page_count=len(reader.pages),
        char_count=len(text),
        filename=filename or "resume.pdf",
        parser="pypdf",
    )


def parse_resume_pdf(file_bytes: bytes, filename: str = "resume.pdf") -> ParsedResume:
    """Extract text from a PDF, using OCR when the file is image-only."""
    if not file_bytes:
        raise ValueError("Empty PDF upload.")

    errors: list[str] = []

    try:
        return _parse_with_pymupdf(file_bytes, filename)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"text-extract/PyMuPDF: {exc}")

    try:
        return _parse_with_pypdf(file_bytes, filename)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"text-extract/pypdf: {exc}")

    try:
        text, page_count = _ocr_pdf_pages(file_bytes)
        if len(text) >= 40:
            return ParsedResume(
                text=text,
                page_count=page_count,
                char_count=len(text),
                filename=filename or "resume.pdf",
                parser="ocr",
            )
        errors.append(f"OCR: got only {len(text)} characters (need >= 40).")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"OCR: {type(exc).__name__}: {exc}")

    raise ValueError(
        "Could not read this PDF. It looks like a scanned/image resume and OCR did not "
        "produce enough text. Try uploading the PNG/JPG version, or a text-based PDF. "
        f"Details: {' | '.join(errors)}"
    )


def parse_resume_image(file_bytes: bytes, filename: str) -> ParsedResume:
    try:
        text = _ocr_image_bytes(file_bytes)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Image OCR failed: {type(exc).__name__}: {exc}") from exc

    if len(text) < 40:
        raise ValueError(
            "Could not read enough text from the image "
            f"({len(text)} characters). Use a clearer scan or a text-based PDF."
        )
    return ParsedResume(
        text=text,
        page_count=1,
        char_count=len(text),
        filename=filename or "resume.png",
        parser="ocr-image",
    )


def parse_resume_bytes(file_bytes: bytes, filename: str) -> ParsedResume:
    """Parse PDF, image, or plain-text resumes."""
    name = (filename or "").lower()
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return parse_resume_image(file_bytes, filename)
    if name.endswith(".pdf") or (len(file_bytes) >= 4 and file_bytes[:4] == b"%PDF"):
        return parse_resume_pdf(file_bytes, filename=filename)
    if name.endswith(".txt") or name.endswith(".md"):
        text = _normalize_whitespace(file_bytes.decode("utf-8", errors="ignore"))
        if len(text) < 40:
            raise ValueError("Text resume is too short to analyze.")
        return ParsedResume(
            text=text,
            page_count=1,
            char_count=len(text),
            filename=filename,
            parser="plaintext",
        )
    if file_bytes[:8].startswith(b"\x89PNG") or file_bytes[:3] == b"\xff\xd8\xff":
        return parse_resume_image(file_bytes, filename or "resume.png")
    raise ValueError("Unsupported file type. Upload a PDF, PNG/JPG, or TXT resume.")

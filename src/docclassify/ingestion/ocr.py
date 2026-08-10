"""
Offline OCR fallback. Only invoked for pages parsers.py flagged as having no
real text layer (see MIN_TEXT_LENGTH_PER_PAGE). Tesseract runs first since it's
CPU-only and fast; EasyOCR is used as a fallback when Tesseract's confidence is
low (messier/rotated scans), since EasyOCR's deep-learning models tend to be
more robust at the cost of needing the GPU.
"""
import fitz
import numpy as np
import cv2
import pytesseract
from pytesseract import Output

from docclassify.config import CONFIG

CONFIDENCE_FLAG_THRESHOLD = CONFIG["ocr"]["confidence_flag_threshold"]

_easyocr_reader = None  # lazy-loaded, only if we actually need the fallback


def _get_easyocr_reader(languages: list[str]):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(languages, gpu=True)
    return _easyocr_reader


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """Deskew, binarize, denoise — standard corrections that meaningfully improve OCR accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Otsu's thresholding gives a clean black/white binarization automatically.
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew: find the dominant text angle and rotate to correct it.
    coords = np.column_stack(np.where(binarized > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = binarized.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        binarized = cv2.warpAffine(binarized, M, (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return binarized


def _page_to_image(pdf_path: str, page_index: int, zoom: float = 2.0) -> np.ndarray:
    """Rasterize one PDF page to an image array at higher-than-default DPI (helps OCR accuracy)."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA -> BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def ocr_page(pdf_path: str, page_index: int, tesseract_lang: str = "eng") -> dict:
    """
    Returns {"text": str, "confidence": float (0-100), "engine": "tesseract"|"easyocr"}.
    Falls back to EasyOCR automatically if Tesseract's average confidence is low.
    """
    raw_img = _page_to_image(pdf_path, page_index)
    processed = _preprocess_image(raw_img)

    data = pytesseract.image_to_data(processed, lang=tesseract_lang, output_type=Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confidences = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and int(c) >= 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    text = " ".join(words)

    if avg_conf >= CONFIDENCE_FLAG_THRESHOLD or not words:
        return {"text": text, "confidence": avg_conf, "engine": "tesseract"}

    # Low confidence -> retry with EasyOCR, which tends to handle noisy/rotated
    # scans better since it's a learned model rather than heuristic-based.
    reader = _get_easyocr_reader([tesseract_lang[:2]])  # crude lang-code mapping; refine per your corpus
    results = reader.readtext(processed, detail=1)
    if not results:
        return {"text": text, "confidence": avg_conf, "engine": "tesseract"}  # keep original if EasyOCR finds nothing
    easy_text = " ".join(r[1] for r in results)
    easy_conf = sum(r[2] for r in results) / len(results) * 100
    if easy_conf > avg_conf:
        return {"text": easy_text, "confidence": easy_conf, "engine": "easyocr"}
    return {"text": text, "confidence": avg_conf, "engine": "tesseract"}


def ocr_flagged_pages(parsed: dict, tesseract_lang: str = "eng") -> dict:
    """
    Takes the dict returned by parsers.parse_pdf(), OCRs every page it flagged
    as text-less, and returns an updated dict with those pages' text filled in
    plus the lowest confidence seen (used to decide whether to flag for human review).
    """
    if not parsed.get("needs_ocr_pages"):
        return parsed

    pdf_path = parsed["pdf_handle_path"]
    pages = list(parsed["pages"])
    min_confidence = 100.0

    for page_index in parsed["needs_ocr_pages"]:
        result = ocr_page(pdf_path, page_index, tesseract_lang=tesseract_lang)
        pages[page_index] = result["text"]
        min_confidence = min(min_confidence, result["confidence"])

    parsed["text"] = "\n\n".join(pages)
    parsed["pages"] = pages
    parsed["ocr_min_confidence"] = min_confidence
    return parsed

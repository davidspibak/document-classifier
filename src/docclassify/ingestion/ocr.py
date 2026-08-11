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

# Skew below this is not worth correcting — rotating resamples the whole page and
# costs a little sharpness, which hurts OCR more than a fraction of a degree does.
MIN_DESKEW_ANGLE_DEGREES = 0.3

_easyocr_reader = None  # lazy-loaded, only if we actually need the fallback


def _get_easyocr_reader(languages: list[str]):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        try:
            _easyocr_reader = easyocr.Reader(languages, gpu=True)
        except Exception:  # noqa: BLE001 - no CUDA / VRAM exhausted / driver mismatch
            # Slower, but a missing GPU shouldn't turn the OCR fallback into a
            # hard failure for the whole document.
            _easyocr_reader = easyocr.Reader(languages, gpu=False)
    return _easyocr_reader


def _deskew_angle(binarized: np.ndarray) -> float:
    """
    Estimates the page's skew in degrees from the minimum-area rectangle around
    the TEXT pixels.

    Two details matter here and are easy to get wrong:
      * After THRESH_BINARY + Otsu the text is black (0) and the background is
        white (255), so the text mask is `== 0`. Selecting `> 0` would measure the
        background — i.e. the whole page — and yield a meaningless angle.
      * cv2.minAreaRect wants (x, y) points as int32/float32. np.where returns
        (row, col) == (y, x) as int64, which OpenCV rejects outright.
    """
    text_pixels = np.column_stack(np.where(binarized == 0))
    if len(text_pixels) == 0:
        return 0.0

    points = text_pixels[:, ::-1].astype(np.int32)  # (y, x) -> (x, y), CV_32S
    angle = cv2.minAreaRect(points)[-1]

    # OpenCV < 4.5 reports this angle in [-90, 0); >= 4.5 reports it in (0, 90].
    # Folding either convention into (-45, 45] makes the correction below
    # version-independent.
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """Deskew, binarize, denoise — standard corrections that meaningfully improve OCR accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Otsu's thresholding gives a clean black/white binarization automatically.
    # Polarity stays "black text on white" because that's what Tesseract expects.
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    angle = _deskew_angle(binarized)
    if abs(angle) < MIN_DESKEW_ANGLE_DEGREES:
        return binarized

    h, w = binarized.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(binarized, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def _page_to_image(doc: "fitz.Document", page_index: int, zoom: float = 2.0) -> np.ndarray:
    """Rasterize one PDF page to an image array at higher-than-default DPI (helps OCR accuracy)."""
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA -> BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:  # grayscale -> BGR, since _preprocess_image expects 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _ocr_image(raw_img: np.ndarray, tesseract_lang: str = "eng") -> dict:
    """
    Returns {"text": str, "confidence": float (0-100), "engine": "tesseract"|"easyocr"}.
    Falls back to EasyOCR automatically if Tesseract's average confidence is low.
    """
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


def ocr_page(pdf_path: str, page_index: int, tesseract_lang: str = "eng") -> dict:
    """Single-page convenience wrapper. Opens and closes the PDF around the one page."""
    with fitz.open(pdf_path) as doc:
        raw_img = _page_to_image(doc, page_index)
    return _ocr_image(raw_img, tesseract_lang=tesseract_lang)


def ocr_flagged_pages(parsed: dict, tesseract_lang: str = "eng") -> dict:
    """
    Takes the dict returned by parsers.parse_pdf(), OCRs every page it flagged
    as text-less, and returns an updated dict with those pages' text filled in
    plus the lowest confidence seen (used to decide whether to flag for human review).

    The PDF is opened ONCE for all flagged pages rather than per page — a scanned
    document can flag every page, and re-parsing the file each time is both slow
    and a file-handle leak.
    """
    if not parsed.get("needs_ocr_pages"):
        return parsed

    pdf_path = parsed["pdf_handle_path"]
    pages = list(parsed["pages"])
    min_confidence = 100.0

    with fitz.open(pdf_path) as doc:
        for page_index in parsed["needs_ocr_pages"]:
            raw_img = _page_to_image(doc, page_index)
            result = _ocr_image(raw_img, tesseract_lang=tesseract_lang)
            pages[page_index] = result["text"]
            min_confidence = min(min_confidence, result["confidence"])

    parsed["text"] = "\n\n".join(pages)
    parsed["pages"] = pages
    parsed["ocr_min_confidence"] = min_confidence
    return parsed

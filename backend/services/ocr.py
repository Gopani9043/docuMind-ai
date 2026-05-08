import platform
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging
import platform

logger = logging.getLogger(__name__)

# Windows-specific configuration
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
else:
    # Linux/macOS use system PATH
    POPPLER_PATH = None


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    try:
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            return _extract_from_pdf(file_bytes)

        elif ext in ["png", "jpg", "jpeg", "tiff", "tif"]:
            return _extract_from_image(file_bytes)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        logger.error(f"OCR failed for {filename}: {e}")
        raise


def _extract_from_pdf(file_bytes: bytes) -> str:
    if POPPLER_PATH:
        images = convert_from_bytes(
            file_bytes,
            dpi=300,
            poppler_path=POPPLER_PATH
        )
    else:
        images = convert_from_bytes(
            file_bytes,
            dpi=300
        )

    all_text = []

    for i, image in enumerate(images):
        logger.info(f"Running OCR on page {i+1}/{len(images)}")

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        all_text.append(text)

    return "\n\n--- PAGE BREAK ---\n\n".join(all_text)


def _extract_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))

    return pytesseract.image_to_string(
        image,
        lang="eng"
    )
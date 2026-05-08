# ── Base image ────────────────────────────────────
FROM python:3.11-slim

# ── Set working directory ─────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────
# Tesseract OCR + Poppler — installed at OS level
# No more manual PATH configuration needed!
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
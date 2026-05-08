# DocParse — Intelligent Document Processing Pipeline

> AI-powered system that extracts structured data from any document automatically.
> Upload an invoice, contract, receipt or report — get clean JSON back instantly.

![Tests](https://github.com/Gopani9043/intelligent-document-processing/actions/workflows/ci.yml/badge.svg)

---

## What it does

DocParse takes any PDF or image document and automatically extracts structured data using OCR and a Large Language Model. No templates, no manual rules — it understands any document layout.

**Supported document types:**
- Invoices → invoice number, vendor, amounts, VAT, line items
- Contracts → parties, dates, value, key clauses
- Receipts → vendor, date, total, items
- Reports → title, author, date, key findings

---

## Benchmark Results

| Document Type | Accuracy | Documents Tested |
|---|---|---|
| Invoices | 100% | 2 |
| Receipts | 100% | 1 |
| Contracts | 100% | 1 |
| Reports | 100% | 1 |
| **Overall** | **100%** | **4** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Python 3.11 |
| OCR Engine | Tesseract 5 + pdf2image |
| LLM | LangChain + Groq (Llama 3.3 70B) |
| Database | PostgreSQL 15 |
| File Storage | MinIO (S3-compatible) |
| Frontend | React + Vite + Tailwind CSS |
| Infrastructure | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## Architecture

```text
┌─────────────────────┐      POST /upload       ┌─────────────────────┐
│   React Frontend    │ ─────────────────────► │   FastAPI Backend   │
│     localhost:80    │ ◄───────────────────── │    localhost:8000   │
└─────────────────────┘      JSON Response     └─────────┬───────────┘
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    │                                     │                                     │
                    ▼                                     ▼                                     ▼

          ┌─────────────────┐                 ┌─────────────────┐                 ┌────────────────────┐
          │  Tesseract OCR  │                 │    Groq LLM     │                 │    PostgreSQL      │
          │   PDF → Text    │                 │   Text → JSON   │                 │   Results Storage  │
          └─────────────────┘                 └─────────────────┘                 └────────────────────┘

                                            ┌────────────────────┐
                                            │      MinIO S3      │
                                            │    File Storage    │
                                            └────────────────────┘
```

**Services:**

| Service | Technology | Port |
|---|---|---|
| Frontend | React + Nginx | 80 |
| Backend API | FastAPI + Uvicorn | 8000 |
| Database | PostgreSQL 15 | 5432 |
| File Storage | MinIO | 9000 |
| MinIO Dashboard | MinIO Console | 9001 |
---

## Quick Start

### Prerequisites
- Docker Desktop
- Groq API key (free at console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/Gopani9043/intelligent-document-processing.git
cd intelligent-document-processing
```

### 2. Create .env file
```bash
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
APP_ENV=development
```

### 3. Start everything
```bash
docker compose up -d
```

### 4. Open the app
- React dashboard → http://localhost
- API docs → http://localhost:8000/docs
- MinIO dashboard → http://localhost:9001

That's it. One command and the full stack is running.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /upload | Upload document → returns doc_id |
| GET | /results/{doc_id} | Get extraction results |
| GET | /documents | List all uploaded documents |
| GET | /benchmark | Get benchmark accuracy results |
| GET | /health | Health check |

### Example

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@invoice.pdf"

# Response
{
  "doc_id": "a3f7c821-4b2e-4d1a-9c3e-1234abcd5678",
  "status": "done",
  "message": "Document processed successfully as invoice"
}

# Get results
curl http://localhost:8000/results/a3f7c821-4b2e-4d1a-9c3e-1234abcd5678
```

---

## Running Tests

```bash
pytest tests/test_ocr.py tests/test_extractor.py tests/test_api.py -v
```

**13 tests — all passing.**



## Project Structure
```text
docparse/
├── backend/
│   ├── database/
│   │   ├── connection.py   # SQLAlchemy async setup
│   │   ├── models.py       # PostgreSQL table definitions
│   │   └── crud.py         # Database operations
│   ├── services/
│   │   ├── ocr.py          # Tesseract OCR pipeline
│   │   ├── extractor.py    # LangChain + Groq LLM
│   │   └── storage.py      # MinIO file storage
│   ├── models/
│   │   └── schemas.py      # Pydantic data models
│   ├── main.py             # FastAPI application
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/          # Upload, Results, Benchmarks
│   │   ├── hooks/          # useUpload hook
│   │   └── App.jsx         # Routing
│   └── Dockerfile
├── tests/
│   ├── test_ocr.py
│   ├── test_extractor.py
│   ├── test_api.py
│   └── benchmark.py
├── sample_documents/       # Example PDFs for testing
├── docker-compose.yml
└── README.md

---

## Sample Documents

The `sample_documents/` folder contains example PDFs you can use to test the system immediately after setup.

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [LangChain](https://langchain.com/) — LLM application framework  
- [Groq](https://groq.com/) — Free LLM inference (Llama 3.3 70B)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Open source OCR
- [MinIO](https://min.io/) — S3-compatible object storage
- [PostgreSQL](https://www.postgresql.org/) — Relational database

---

*Built to explore AI-powered document processing using OCR, LLMs, and modern backend infrastructure.*
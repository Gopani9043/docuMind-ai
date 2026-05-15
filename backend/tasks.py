import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import uuid
from celery_app import celery
from services.ocr import extract_text_from_file
from services.extractor import extract_structured_data
from services.storage import upload_file
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Use SYNC database connection for Celery (not async)
SYNC_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://docparse:docparse123@localhost:5432/docparse_db"
).replace("postgresql+asyncpg://", "postgresql+psycopg2://")

sync_engine = create_engine(SYNC_DB_URL)
SyncSession = sessionmaker(sync_engine)


@celery.task(bind=True, name="process_document")
def process_document(self, file_bytes_hex, filename, content_type, doc_id):
    logger.info(f"[{doc_id}] Task started")
    db = SyncSession()
    uid = uuid.UUID(doc_id)

    try:
        file_bytes = bytes.fromhex(file_bytes_hex)

        # Upload to MinIO
        logger.info(f"[{doc_id}] Uploading to MinIO...")
        s3_key = upload_file(file_bytes, filename, content_type, doc_id)

        # Update s3_key
        db.execute(
            text("UPDATE documents SET s3_key=:key WHERE id=:id"),
            {"key": s3_key, "id": uid}
        )
        db.commit()

        # OCR
        logger.info(f"[{doc_id}] Running OCR...")
        raw_text = extract_text_from_file(file_bytes, filename)

        # LLM extraction
        logger.info(f"[{doc_id}] Running LLM extraction...")
        extraction = extract_structured_data(raw_text)

        # Save result
        import json
        result_id = uuid.uuid4()
        db.execute(text("""
            INSERT INTO extraction_results 
            (id, doc_id, document_type, extracted_data, confidence, raw_text, created_at)
            VALUES (:id, :doc_id, :doc_type, :data, :conf, :raw, NOW())
        """), {
            "id": result_id,
            "doc_id": uid,
            "doc_type": extraction.get("document_type", "unknown"),
            "data": json.dumps(extraction.get("extracted_data", {})),
            "conf": json.dumps(extraction.get("confidence", {})),
            "raw": raw_text
        })

        # Update status to done
        db.execute(
            text("UPDATE documents SET status='done' WHERE id=:id"),
            {"id": uid}
        )
        db.commit()
        logger.info(f"[{doc_id}] Done!")

    except Exception as e:
        logger.error(f"[{doc_id}] Failed: {e}")
        db.rollback()
        try:
            import json
            db.execute(text("""
                INSERT INTO extraction_results
                (id, doc_id, document_type, extracted_data, confidence, error, created_at)
                VALUES (:id, :doc_id, 'unknown', '{}', '{}', :error, NOW())
            """), {"id": uuid.uuid4(), "doc_id": uid, "error": str(e)})
            db.execute(
                text("UPDATE documents SET status='error' WHERE id=:id"),
                {"id": uid}
            )
            db.commit()
        except Exception:
            db.rollback()
        raise
    finally:
        db.close()
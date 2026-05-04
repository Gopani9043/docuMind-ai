import os
import io
import logging
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── MinIO client setup ────────────────────────────
client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
    secure=os.getenv("MINIO_SECURE", "False").lower() == "true"
)

BUCKET_NAME = os.getenv("MINIO_BUCKET", "docparse-documents")


def init_storage() -> None:
    """
    Called once on app startup.
    Creates the bucket if it does not exist yet.
    """
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            logger.info(f"Created MinIO bucket: {BUCKET_NAME}")
        else:
            logger.info(f"MinIO bucket already exists: {BUCKET_NAME}")
    except S3Error as e:
        logger.error(f"MinIO bucket init failed: {e}")
        raise


def upload_file(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    doc_id: str
) -> str:
    """
    Upload a file to MinIO.
    Returns the s3_key (the path inside the bucket).

    Files are stored as:
    documents/2024-10-15/doc_id/filename.pdf
    """
    from datetime import datetime
    date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
    s3_key = f"documents/{date_prefix}/{doc_id}/{filename}"

    try:
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=s3_key,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type
        )
        logger.info(f"Uploaded to MinIO: {s3_key}")
        return s3_key

    except S3Error as e:
        logger.error(f"MinIO upload failed: {e}")
        raise


def get_file_url(s3_key: str, expires_hours: int = 1) -> str:
    """
    Generate a temporary download URL for a stored file.
    URL expires after expires_hours (default 1 hour).
    Useful for letting users re-download their original file.
    """
    from datetime import timedelta
    try:
        url = client.presigned_get_object(
            bucket_name=BUCKET_NAME,
            object_name=s3_key,
            expires=timedelta(hours=expires_hours)
        )
        return url
    except S3Error as e:
        logger.error(f"Failed to generate URL: {e}")
        raise


def delete_file(s3_key: str) -> None:
    """
    Delete a file from MinIO by its s3_key.
    Used for cleanup if processing fails badly.
    """
    try:
        client.remove_object(BUCKET_NAME, s3_key)
        logger.info(f"Deleted from MinIO: {s3_key}")
    except S3Error as e:
        logger.error(f"MinIO delete failed: {e}")
        raise
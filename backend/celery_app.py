import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery = Celery(
    "docparse",
    broker=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
    include=["tasks"]
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Berlin",
    enable_utc=True,
    task_track_started=True,
)
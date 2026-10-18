"""The Celery application: the broker connection shared by the API (which only
sends tasks) and the worker (which runs them)."""

from celery import Celery

from app.core.config import get_settings

# Task names. The API sends tasks by name, so it never imports worker code.
PROCESS_DOCUMENT = "documind.process_document"
SWEEP = "documind.sweep"

celery_app = Celery("documind", broker=get_settings().REDIS_URL)
celery_app.conf.update(
    # We keep job state in Postgres; Celery's result storage isn't needed.
    task_ignore_result=True,
    broker_transport_options={
        # Fail fast when Redis is down instead of hanging an API request.
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
    },
    broker_connection_retry_on_startup=True,
)

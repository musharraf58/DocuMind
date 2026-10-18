"""The Celery application: the broker connection shared by the API (which only
sends tasks) and the worker (which runs them).

Start the worker (the compose "worker" service does this):

    celery -A app.workers.celery_app worker --beat --concurrency=1
"""

from celery import Celery
from celery.signals import after_setup_logger, worker_ready

from app.core.config import get_settings
from app.core.logging import configure_logging

# Task names. The API sends tasks by name, so it never imports worker code.
PROCESS_DOCUMENT = "documind.process_document"
SWEEP = "documind.sweep"

settings = get_settings()
job_seconds = int(settings.JOB_TIMEOUT_MINUTES * 60)

celery_app = Celery("documind", broker=settings.REDIS_URL, include=["app.workers.tasks"])
celery_app.conf.update(
    # We keep job state in Postgres; Celery's result storage isn't needed.
    task_ignore_result=True,
    # One job at a time, and don't reserve more than that: the embedding rate
    # limit is the bottleneck, so parallel jobs would only queue behind it.
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    # Remove a job from the queue only after it finished. If the worker dies
    # mid-job, the job is delivered again ("at least once"), which is safe
    # because the job is idempotent.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # The job enforces JOB_TIMEOUT_MINUTES itself (cleanly, as a retryable
    # failure). This hard limit is only a backstop for a job that hangs.
    task_time_limit=job_seconds + 300,
    broker_transport_options={
        # Redis hands an unacknowledged job to another worker after this long,
        # so it must exceed the longest job, or a long job would run twice.
        "visibility_timeout": job_seconds + 1800,
        # Fail fast when Redis is down instead of hanging an API request.
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
    },
    broker_connection_retry_on_startup=True,
    # Celery beat (started with the worker via --beat) runs the sweeper.
    beat_schedule={"sweep": {"task": SWEEP, "schedule": 60.0}},
    worker_hijack_root_logger=False,
)


@after_setup_logger.connect
def show_app_logs(**kwargs) -> None:
    """Without this, the job's and the sweeper's INFO messages would be dropped:
    the logging setup in app.main only runs in the API process."""
    configure_logging()


@worker_ready.connect
def sweep_at_startup(sender=None, **kwargs) -> None:
    """Catch up right away on anything left over while the worker was down."""
    celery_app.send_task(SWEEP)

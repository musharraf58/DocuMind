import json
import logging

from app.core.request_id import get_request_id

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] - %(message)s"


class RequestIdFilter(logging.Filter):
    """Adds record.request_id ("-" outside a request) to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log collectors. Standard library only."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(log_format: str | None = None) -> None:
    """Show the app's INFO messages (job progress, sweeper actions, embedding
    batches) with timestamps and the request id; libraries stay at WARNING.
    Used by both the API and the worker, so their logs look the same.
    log_format: "text" or "json" (default: the LOG_FORMAT setting)."""
    if log_format is None:
        from app.core.config import get_settings

        log_format = get_settings().LOG_FORMAT
    logging.basicConfig(level=logging.WARNING)
    formatter = JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
        # On the handler, not a logger: records from every logger (ours,
        # uvicorn's, Celery's) pass through the root handlers.
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())
    logging.getLogger("app").setLevel(logging.INFO)

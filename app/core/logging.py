import json
import logging

from app.core.request_id import get_request_id

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] - %(message)s"
# Loggers that carry their own handlers (and don't pass records on to the root
# logger): uvicorn's server and access log, Celery's task log. Their handlers
# get our formatter, or their lines would have no request id.
OWN_HANDLER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "celery.task")

_original_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs) -> logging.LogRecord:
    """Every log record, from any logger and any handler, gets request_id
    ("-" outside a request) the moment it is created. A filter on one
    handler would miss records that go to other handlers."""
    record = _original_factory(*args, **kwargs)
    record.request_id = get_request_id() or "-"
    return record


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


def make_formatter(log_format: str) -> logging.Formatter:
    return JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT)


def apply_formatter(logger: logging.Logger, formatter: logging.Formatter) -> None:
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def configure_logging(log_format: str | None = None) -> None:
    """Show the app's INFO messages (job progress, sweeper actions, embedding
    batches) with timestamps and the request id; libraries stay at WARNING.
    Used by both the API and the worker, so their logs look the same.
    log_format: "text" or "json" (default: the LOG_FORMAT setting)."""
    if log_format is None:
        from app.core.config import get_settings

        log_format = get_settings().LOG_FORMAT
    logging.setLogRecordFactory(_record_factory)
    logging.basicConfig(level=logging.WARNING)
    formatter = make_formatter(log_format)
    apply_formatter(logging.getLogger(), formatter)
    for name in OWN_HANDLER_LOGGERS:
        apply_formatter(logging.getLogger(name), formatter)
    logging.getLogger("app").setLevel(logging.INFO)

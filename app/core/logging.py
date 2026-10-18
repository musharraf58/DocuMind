import logging


def configure_logging() -> None:
    """Show the app's INFO messages (job progress, sweeper actions, embedding
    batches) with timestamps; libraries stay at WARNING. Used by both the API
    and the worker, so their logs look the same."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    logging.getLogger("app").setLevel(logging.INFO)

import math


class LLMError(Exception):
    """Generating an answer failed. The message may reach users and logs, so it
    must be short and never contain the API key or the provider's response."""


class RetryableLLMError(LLMError):
    """May succeed if tried again later: rate limit, timeout, provider 5xx.
    retry_after: the provider's own hint, in seconds, if it sent one."""

    def __init__(
        self, message: str, retry_after: float | None = None, rate_limited: bool = False
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limited = rate_limited


class LLMBusyError(LLMError):
    """No capacity now, and waiting would take longer than an interactive
    request should: tell the client when to come back."""

    def __init__(self, retry_after: float, message: str | None = None) -> None:
        self.retry_after = retry_after
        seconds = max(1, math.ceil(retry_after))
        super().__init__(message or f"The answer service is busy; retry in about {seconds} seconds")


class LLMRefusalError(LLMError):
    """The provider declined to answer (e.g. a safety filter). Not an outage."""

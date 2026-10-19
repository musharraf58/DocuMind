"""Google Gemini over its REST API (generateContent).

https://ai.google.dev/api/generate-content
"""

import logging
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from pydantic import SecretStr

from app.services.llm.base import LLMProvider, LLMResult
from app.services.llm.errors import LLMBusyError, LLMError, LLMRefusalError, RetryableLLMError

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
PACIFIC = ZoneInfo("America/Los_Angeles")  # where Google resets daily quotas
RETRYABLE_STATUSES = {500, 502, 503, 504}
REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

# Fixed, user-safe messages: Google's own error text is never passed on.
UNAVAILABLE = "The answer service is unavailable; try again later"
MESSAGES = {
    400: "The answer service rejected the request",
    401: "The answer service rejected the API key",
    403: "The answer service rejected the API key",
    404: "The answer model was not found; check LLM_MODEL",
}


class GeminiLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: SecretStr,
        timeout: float = 30,
        temperature: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._transport = transport  # tests pass a MockTransport; None means the real network

    async def _generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        key = self._api_key.get_secret_value()
        if not key:
            raise LLMError("The answer service API key is not configured")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            # No thinking setting: Flash-Lite's default is already "minimal".
            # Thinking tokens count toward maxOutputTokens.
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": self._temperature},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{API_BASE}/{self.model}:generateContent",
                    json=body,
                    # In a header, never in the URL: request URLs end up in logs.
                    headers={"x-goog-api-key": key},
                )
        except httpx.TimeoutException as exc:
            raise RetryableLLMError("The answer service timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableLLMError("Could not reach the answer service") from exc

        if response.status_code != 200:
            self._raise_for_status(response)
        return self._parse(response)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        error = _error_details(response)
        # Status and Google's error code only: the message may quote the request.
        logger.warning("Gemini request failed with HTTP %d (%s)", status, error.get("status", "?"))
        if status == 429:
            if _daily_quota_exhausted(error):
                seconds = _seconds_until_pacific_midnight()
                hours = max(1, math.ceil(seconds / 3600))
                raise LLMBusyError(
                    seconds,
                    "The answer service's daily limit is used up; it resets at midnight "
                    f"Pacific time (in about {hours} hours)",
                )
            raise RetryableLLMError(
                "The answer service rate limit was reached",
                retry_after=_retry_after(response, error),
                rate_limited=True,
            )
        if status in RETRYABLE_STATUSES:
            raise RetryableLLMError(UNAVAILABLE, retry_after=_retry_after(response, error))
        raise LLMError(MESSAGES.get(status, f"Answer request failed (HTTP {status})"))

    def _parse(self, response: httpx.Response) -> LLMResult:
        try:
            data = response.json()
            if (data.get("promptFeedback") or {}).get("blockReason"):
                raise LLMRefusalError("The answer service declined to answer this question")
            candidates = data.get("candidates") or []
            if not candidates:
                raise LLMError("The answer service returned an unexpected response")
            candidate = candidates[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            # Thought summaries (if any) are not part of the answer.
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            reason = candidate.get("finishReason")
            usage = data.get("usageMetadata") or {}
            prompt_tokens = int(usage.get("promptTokenCount") or 0)
            # Thinking tokens are billed as output, so count them as completion.
            completion_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
                usage.get("thoughtsTokenCount") or 0
            )
        except LLMError:
            raise
        except (ValueError, TypeError, AttributeError) as exc:
            raise LLMError("The answer service returned an unexpected response") from exc

        if not text:
            if reason in REFUSAL_REASONS:
                raise LLMRefusalError("The answer service declined to answer this question")
            raise LLMError("The answer service returned an empty answer")
        if reason == "MAX_TOKENS":
            logger.warning("Gemini answer was cut off at LLM_MAX_OUTPUT_TOKENS")
        return LLMResult(text, prompt_tokens, completion_tokens)


def _error_details(response: httpx.Response) -> dict:
    try:
        error = response.json().get("error")
    except ValueError:
        return {}
    return error if isinstance(error, dict) else {}


def _retry_after(response: httpx.Response, error: dict) -> float | None:
    """The Retry-After header, else Google's RetryInfo detail ("retryDelay": "31s")."""
    try:
        return max(0.0, float(response.headers["Retry-After"]))
    except (KeyError, ValueError):
        pass
    for detail in error.get("details") or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, float(delay[:-1]))
            except ValueError:
                return None
    return None


def _daily_quota_exhausted(error: dict) -> bool:
    """Google's QuotaFailure detail names the quota that ran out; daily ones
    contain "PerDay" (e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier)."""
    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        for violation in detail.get("violations") or []:
            quota = str(violation.get("quotaId", "")) + str(violation.get("quotaMetric", ""))
            if "PerDay" in quota or "per_day" in quota:
                return True
    return False


def _seconds_until_pacific_midnight(now: datetime | None = None) -> float:
    now = (now or datetime.now(PACIFIC)).astimezone(PACIFIC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - now).total_seconds()

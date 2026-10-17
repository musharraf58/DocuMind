import logging

import httpx
from pydantic import SecretStr

from app.services.embeddings.base import BatchResult, EmbeddingProvider, InputType
from app.services.embeddings.errors import EmbeddingError, RetryableEmbeddingError

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
# Total tokens allowed per request, from Voyage's docs (checked 2026-10).
# Models not listed get the smallest documented limit.
TOKEN_LIMITS = {
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
}
DEFAULT_TOKEN_LIMIT = 120_000
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Fixed, user-safe messages. Voyage's own response text is never passed on:
# it could echo input or request details back.
UNAVAILABLE = "Embedding service is unavailable; try again later"
MESSAGES = {
    400: "Embedding service rejected the input",
    401: "Embedding service rejected the API key",
    403: "Embedding service refused the request",
    429: "Embedding service rate limit reached; try again later",
}


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI over its REST API (https://docs.voyageai.com/reference/embeddings-api)."""

    max_batch_texts = 1000

    def __init__(
        self,
        api_key: SecretStr,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_batch_tokens = TOKEN_LIMITS.get(self.model, DEFAULT_TOKEN_LIMIT)
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport  # tests pass a MockTransport; None means the real network

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        key = self._api_key.get_secret_value()
        if not key:
            raise EmbeddingError("Embedding API key is not configured")
        body = {
            "input": texts,
            "model": self.model,
            # Voyage prepends a different retrieval prompt for queries and documents.
            "input_type": input_type,
            "output_dimension": self.dimensions,
            # Error instead of silently embedding a cut-off text.
            "truncation": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    VOYAGE_URL, json=body, headers={"Authorization": f"Bearer {key}"}
                )
        except httpx.TimeoutException as exc:
            raise RetryableEmbeddingError("Embedding service timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableEmbeddingError("Could not reach the embedding service") from exc

        if response.status_code != 200:
            # Status only: the body may contain input text.
            logger.warning("Voyage embeddings request failed with HTTP %d", response.status_code)
            message = MESSAGES.get(response.status_code, UNAVAILABLE)
            if response.status_code in RETRYABLE_STATUSES:
                raise RetryableEmbeddingError(message, _retry_after(response))
            if response.status_code not in MESSAGES:
                message = f"Embedding request failed (HTTP {response.status_code})"
            raise EmbeddingError(message)

        try:
            payload = response.json()
            items = sorted(payload["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in items]
            tokens = int(payload.get("usage", {}).get("total_tokens", 0))
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Embedding service returned an unexpected response") from exc
        return BatchResult(vectors, tokens)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None  # missing, or an HTTP date (rare); fall back to our backoff

from collections.abc import Callable, Sequence

from app.services.chunking import count_tokens
from app.services.llm.base import LLMProvider, LLMResult

DEFAULT_ANSWER = "Here is what the sources say [1]."


class FakeLLMProvider(LLMProvider):
    """Scripted answers, no network. Records every call so tests can check
    what was sent, or that nothing was sent at all.

    answer: fixed text, or a function (system, user) -> text.
    failures: exceptions to raise on the first calls, in order.
    """

    def __init__(
        self,
        answer: str | Callable[[str, str], str] = DEFAULT_ANSWER,
        failures: Sequence[Exception] = (),
        **kwargs,
    ) -> None:
        kwargs.setdefault("model", "fake-llm")
        super().__init__(**kwargs)
        self.answer = answer
        self.failures = list(failures)
        self.calls: list[tuple[str, str, int]] = []

    async def _generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        self.calls.append((system, user, max_tokens))
        if self.failures:
            raise self.failures.pop(0)
        text = self.answer(system, user) if callable(self.answer) else self.answer
        return LLMResult(
            text=text,
            prompt_tokens=count_tokens(system) + count_tokens(user),
            completion_tokens=count_tokens(text),
        )

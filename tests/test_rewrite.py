"""Rewriting follow-ups into standalone questions (best-effort)."""

import asyncio

from app.services.conversations import HistoryTurn
from app.services.llm import FakeLLMProvider, LLMBusyError, LLMError
from app.services.rewrite import REWRITE_MAX_TOKENS, REWRITE_SYSTEM, rewrite_question

HISTORY = [HistoryTurn("What is the refund policy?", "Refunds take 30 days [1].")]


async def rewrite(llm, question="And for damaged items?", history=HISTORY, **kwargs):
    kwargs.setdefault("max_seconds", 10)
    kwargs.setdefault("max_chars", 1000)
    return await rewrite_question(llm, question, history, **kwargs)


async def test_first_question_is_never_rewritten():
    llm = FakeLLMProvider()
    result = await rewrite(llm, "What is the refund policy?", history=[])
    assert (result.question, result.rewritten, result.reason) == (
        "What is the refund policy?",
        False,
        "first question",
    )
    assert llm.calls == []


async def test_follow_up_is_rewritten_with_one_short_call():
    llm = FakeLLMProvider(answer="What is the refund policy for damaged items?\n")
    result = await rewrite(llm)

    assert result.rewritten and result.question == "What is the refund policy for damaged items?"
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    [(system, user, max_tokens)] = llm.calls
    assert system == REWRITE_SYSTEM and max_tokens == REWRITE_MAX_TOKENS
    assert "<history>" in user and "What is the refund policy?" in user
    assert user.rstrip().endswith("<question>\nAnd for damaged items?\n</question>")
    assert "[1]" not in user  # old citation markers removed


async def test_quotes_and_extra_lines_are_cleaned():
    llm = FakeLLMProvider(answer='\n"What about damaged items?"\nSome explanation.')
    assert (await rewrite(llm)).question == "What about damaged items?"


async def test_failure_falls_back_to_the_original_question():
    for error in (LLMError("rejected"), LLMBusyError(retry_after=30), RuntimeError("bug")):
        llm = FakeLLMProvider(failures=[error])
        result = await rewrite(llm)
        assert (result.question, result.rewritten) == ("And for damaged items?", False)
        assert result.reason.startswith("failed")


async def test_timeout_falls_back_to_the_original_question():
    class Slow(FakeLLMProvider):
        async def _generate(self, system, user, max_tokens):
            await asyncio.sleep(5)

    result = await rewrite(Slow(), max_seconds=0.05)
    assert (result.question, result.rewritten, result.reason) == (
        "And for damaged items?",
        False,
        "timed out",
    )


async def test_unusable_reply_falls_back():
    for reply in ("   \n  ", "x" * 50):
        result = await rewrite(FakeLLMProvider(answer=reply), max_chars=20)
        assert (result.question, result.rewritten, result.reason) == (
            "And for damaged items?",
            False,
            "unusable reply",
        )

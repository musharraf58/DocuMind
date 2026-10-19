"""Checking the [n] citations in a generated answer.

The model is told to cite sources as [n], but nothing forces it to: it may
cite a number we never supplied, or cite nothing. So citations are parsed
and validated against the sources actually sent, never trusted as given.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.context import Source

# "[2]", "[1, 3]", "[1,3]". At most three digits, so a bracketed year such as
# "[2024]" is left alone rather than treated as a citation.
_CITATION = re.compile(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([.,;:!?])")
_EXTRA_SPACES = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class CheckedAnswer:
    text: str  # the answer with citations of unknown sources removed
    cited: list[Source]  # valid cited sources, in order of first mention
    grounded: bool  # True if the answer cites at least one real source


def check_citations(answer: str, sources: Sequence[Source]) -> CheckedAnswer:
    by_number = {source.number: source for source in sources}
    cited: list[int] = []

    def keep_valid(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = [n for n in numbers if n in by_number]
        for number in valid:
            if number not in cited:
                cited.append(number)
        return f"[{', '.join(str(n) for n in valid)}]" if valid else ""

    text = _CITATION.sub(keep_valid, answer)
    # Tidy the gaps left where an invalid marker was removed.
    text = _EXTRA_SPACES.sub(" ", _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)).strip()
    return CheckedAnswer(text, [by_number[n] for n in cited], grounded=bool(cited))

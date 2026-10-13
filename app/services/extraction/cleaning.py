import re

# Control characters other than \t and \n. Includes \x00, which PostgreSQL
# refuses to store in a text column.
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
# Invisible characters that only get in the way of search: zero-width spaces
# and joiners, byte-order marks, soft hyphens.
_INVISIBLE = re.compile(r"[​-‍⁠﻿­]")
# Unusual spaces (non-breaking, thin, ideographic, ...) become a plain space.
_ODD_SPACES = re.compile(r"[   -   　]")
# A run of spaces/tabs after some text. Leading indentation is left alone so
# code blocks in Markdown keep their shape.
_INNER_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
# letter, hyphen, line break (optionally indented), next letter.
_HYPHEN_BREAK = re.compile(r"([^\W\d_])-\n[ \t]*([^\W\d_])")
_BLANK_LINES = re.compile(r"\n{3,}")


def _rejoin(match: re.Match[str]) -> str:
    # Only rejoin when the next line continues in lowercase ("exam-\nple"), so
    # names like "Jean-\nPaul" keep their hyphen.
    first, second = match.group(1), match.group(2)
    return first + second if second.islower() else f"{first}-{second}"


def clean_text(text: str) -> str:
    """Normalize extracted text so it is safe to store and pleasant to chunk."""
    # Every kind of line break becomes "\n" (form feeds often mark PDF page ends).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\f\v ]", "\n", text).replace(" ", "\n\n")
    text = _CONTROL.sub("", text)
    text = _INVISIBLE.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = _INNER_SPACES.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _HYPHEN_BREAK.sub(_rejoin, text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()

"""Every relative link in README.md and docs/ leads somewhere: the file
exists, and a #fragment matches a heading in it (as GitHub makes anchors).
Also: the environment-variable reference in docs/OPERATIONS.md lists every
setting."""

import re
from pathlib import Path

import pytest

from app.core.config import Settings
from tests.repo_files import ROOT, repo_file

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
INLINE_CODE = re.compile(r"`[^`]*`")


def markdown_files() -> list[Path]:
    return [repo_file("README.md"), *sorted(repo_file("docs").glob("*.md"))]


def prose_lines(path: Path):
    """(line number, text) outside fenced code blocks, inline code removed."""
    fenced = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            yield number, INLINE_CODE.sub("", line)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped,
    spaces become hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # [text](url) -> text
    text = text.replace("`", "").replace("*", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    found, seen = set(), {}
    fenced = False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and (match := re.match(r"#{1,6} (.+)", line)):
            base = slug(match.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    return found


def test_slug_matches_github():
    assert slug("Outbound limits: the provider") == "outbound-limits-the-provider"
    assert slug("Search (debugging view)") == "search-debugging-view"
    assert slug("6. The client address behind Render") == "6-the-client-address-behind-render"
    assert slug("`/ask` and the `done` event") == "ask-and-the-done-event"


def relative_links():
    for path in markdown_files():
        for number, line in prose_lines(path):
            for target in LINK.findall(line):
                if re.match(r"[a-z][a-z0-9+.-]*:", target):  # https:, mailto:
                    continue
                yield pytest.param(path, target, id=f"{path.relative_to(ROOT)}:{number}:{target}")


@pytest.mark.parametrize(("source", "target"), list(relative_links()))
def test_relative_link_resolves(source: Path, target: str):
    file_part, _, fragment = target.partition("#")
    destination = (source.parent / file_part).resolve() if file_part else source
    assert destination.exists(), f"{target}: no such file"
    if fragment:
        assert destination.suffix == ".md", f"{target}: an anchor into a non-Markdown file"
        assert fragment in anchors(destination), f"{target}: no heading with that anchor"


def test_there_are_links_to_check():
    assert len(list(relative_links())) > 30


def test_operations_lists_every_setting():
    reference = repo_file("docs").joinpath("OPERATIONS.md").read_text()
    missing = [name for name in Settings.model_fields if f"| `{name}` |" not in reference]
    assert missing == [], "add these to docs/OPERATIONS.md, Environment variables"

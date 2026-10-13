from dataclasses import replace

import pytest

from app.services.extraction import ExtractionError, ExtractionLimits, Segment, extract

LIMITS = ExtractionLimits(
    max_pages=50,
    max_chars=100_000,
    min_chars=10,
    max_docx_uncompressed_bytes=5 * 1024 * 1024,
    max_docx_entries=100,
)
FILLER = "This sentence is long enough to pass the minimum text check."


# --- TXT ---


def test_txt_is_one_untitled_segment():
    assert extract(FILLER.encode(), "text/plain", LIMITS) == [Segment(FILLER)]


def test_txt_null_characters_are_stripped():
    [segment] = extract(f"null\x00 bytes {FILLER}".encode(), "text/plain", LIMITS)
    assert "\x00" not in segment.text
    assert segment.text.startswith("null bytes")


def test_txt_over_char_limit_fails():
    limits = replace(LIMITS, max_chars=20)
    with pytest.raises(ExtractionError, match="more than 20 characters"):
        extract(FILLER.encode(), "text/plain", limits)


def test_whitespace_only_text_fails_with_readable_message():
    with pytest.raises(ExtractionError, match="No readable text"):
        extract(b"   \n\n\t  ", "text/plain", LIMITS)


def test_unknown_content_type_fails():
    with pytest.raises(ExtractionError):
        extract(b"data", "application/zip", LIMITS)


# --- Markdown ---


def test_markdown_headings_become_section_titles():
    md = f"Intro text. {FILLER}\n\n# First\n\nAlpha body.\n\n## Second ##\n\nBeta body.\n"
    segments = extract(md.encode(), "text/markdown", LIMITS)
    assert [(s.section_title, s.text) for s in segments] == [
        (None, f"Intro text. {FILLER}"),
        ("First", "Alpha body."),
        ("Second", "Beta body."),
    ]
    assert all(s.page_number is None for s in segments)


def test_markdown_hash_inside_code_fence_is_not_a_heading():
    md = f"# Setup\n\n```bash\n# install deps\nuv sync\n```\n\n{FILLER}\n"
    [segment] = extract(md.encode(), "text/markdown", LIMITS)
    assert segment.section_title == "Setup"
    assert "# install deps" in segment.text


def test_markdown_empty_sections_are_dropped():
    md = f"# Empty\n\n# Full\n\n{FILLER}\n"
    segments = extract(md.encode(), "text/markdown", LIMITS)
    assert [s.section_title for s in segments] == ["Full"]


def test_markdown_hashtag_without_space_is_text():
    md = f"#notaheading {FILLER}"
    [segment] = extract(md.encode(), "text/markdown", LIMITS)
    assert segment.section_title is None

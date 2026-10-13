from dataclasses import replace

import pytest

from app.services.extraction import ExtractionError, ExtractionLimits, Segment, extract
from tests.fixtures import build_pdf, encrypt_pdf

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


# --- PDF ---


def test_pdf_pages_keep_their_page_numbers():
    pdf = build_pdf([["Page one says hello."], ["Page two says goodbye."], ["Third page here."]])
    segments = extract(pdf, "application/pdf", LIMITS)
    assert [(s.page_number, s.text) for s in segments] == [
        (1, "Page one says hello."),
        (2, "Page two says goodbye."),
        (3, "Third page here."),
    ]
    assert all(s.section_title is None for s in segments)


def test_pdf_blank_page_is_skipped_but_numbering_holds():
    pdf = build_pdf([["Before the blank page."], [], ["After the blank page."]])
    segments = extract(pdf, "application/pdf", LIMITS)
    assert [s.page_number for s in segments] == [1, 3]


def test_scanned_pdf_with_no_text_layer_fails_with_readable_message():
    with pytest.raises(ExtractionError, match="scanned"):
        extract(build_pdf([[], []]), "application/pdf", LIMITS)


def test_pdf_over_page_limit_fails():
    pdf = build_pdf([[f"Page {n}"] for n in range(3)])
    with pytest.raises(ExtractionError, match="3 pages; the limit is 2"):
        extract(pdf, "application/pdf", replace(LIMITS, max_pages=2))


def test_pdf_over_char_limit_fails():
    pdf = build_pdf([[FILLER]] * 3)
    with pytest.raises(ExtractionError, match="more than 100 characters"):
        extract(pdf, "application/pdf", replace(LIMITS, max_chars=100))


def test_corrupt_pdf_fails_cleanly():
    with pytest.raises(ExtractionError, match="not a readable PDF"):
        extract(b"%PDF-1.4\nthis is not really a pdf", "application/pdf", LIMITS)


def test_password_protected_pdf_fails_with_readable_message():
    pdf = encrypt_pdf(build_pdf([[FILLER]]), user_password="secret")
    with pytest.raises(ExtractionError, match="password-protected"):
        extract(pdf, "application/pdf", LIMITS)


def test_pdf_encrypted_with_empty_password_still_reads():
    pdf = encrypt_pdf(build_pdf([[FILLER]]), user_password="")
    [segment] = extract(pdf, "application/pdf", LIMITS)
    assert segment.text == FILLER

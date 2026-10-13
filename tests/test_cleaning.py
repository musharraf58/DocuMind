from app.services.extraction.cleaning import clean_text


def test_strips_null_and_control_characters():
    assert clean_text("a\x00b\x07c\x1fd") == "abcd"


def test_keeps_tabs_and_newlines_inside_text():
    assert clean_text("a\tb\nc") == "a\tb\nc"


def test_normalizes_line_endings():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_collapses_runs_of_spaces_but_keeps_indentation():
    assert clean_text("x\n    indented   code  here") == "x\n    indented code here"


def test_odd_unicode_spaces_become_plain_spaces():
    assert clean_text("a b c") == "a b c"


def test_removes_invisible_characters():
    assert clean_text("﻿zero​width") == "zerowidth"


def test_trims_trailing_whitespace_on_each_line():
    assert clean_text("a   \nb\t\n") == "a\nb"


def test_collapses_repeated_blank_lines():
    assert clean_text("a\n\n\n\n\nb\n \n \n \nc") == "a\n\nb\n\nc"


def test_rejoins_words_hyphenated_across_lines():
    assert clean_text("an exam-\nple of hyphen-\n  ation") == "an example of hyphenation"


def test_keeps_hyphen_before_capitalized_word():
    assert clean_text("Jean-\nPaul") == "Jean-Paul"


def test_does_not_join_numbers_or_list_items():
    assert clean_text("2020-\n2021") == "2020-\n2021"
    assert clean_text("intro-\n- item") == "intro-\n- item"


def test_whitespace_only_becomes_empty():
    assert clean_text(" \n\x00\t\n ") == ""

import pytest

from ai_search.chunking import chunk_text


def test_chunk_text_returns_empty_for_blank_text() -> None:
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_splits_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunk_text_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)
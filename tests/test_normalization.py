from app.ingestion.normalization import normalize_text


def test_joins_words_broken_by_line_hyphenation():
    assert normalize_text("сотруд-\nника") == "сотрудника"


def test_single_newline_is_not_a_paragraph_break():
    assert normalize_text("первая строка\nвторая строка") == "первая строка вторая строка"


def test_paragraph_breaks_are_preserved():
    assert normalize_text("абзац один\n\n\n\nабзац два") == "абзац один\n\nабзац два"


def test_collapses_special_spaces_and_soft_hyphens():
    assert normalize_text("а­  б\t\tв") == "а б в"

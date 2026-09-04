from app.rag.sparse import HashedSparseVectorizer, tokenize


def test_tokenize_splits_words_and_numbers_and_lowercases():
    assert tokenize("Статья 152-ФЗ о персональных данных") == [
        "статья", "152", "фз", "о", "персональных", "данных",
    ]


def test_empty_text_produces_empty_vector():
    vector = HashedSparseVectorizer().vectorize("")
    assert vector.indices == []
    assert vector.values == []


def test_indices_are_sorted_and_deduplicated():
    vector = HashedSparseVectorizer().vectorize("отпуск отпуск отпуск перенос")
    assert vector.indices == sorted(set(vector.indices))
    assert len(vector.indices) == len(set(vector.indices))


def test_repeated_terms_get_higher_but_sublinear_weight():
    once = HashedSparseVectorizer().vectorize("отпуск")
    thrice = HashedSparseVectorizer().vectorize("отпуск отпуск отпуск")

    assert thrice.values[0] > once.values[0]
    # Сублинейно: вес утраивается не втрое, а куда меньше.
    assert thrice.values[0] < once.values[0] * 3


def test_same_text_is_vectorized_deterministically():
    vectorizer = HashedSparseVectorizer()
    first = vectorizer.vectorize("отпуск предоставляется ежегодно")
    second = vectorizer.vectorize("отпуск предоставляется ежегодно")

    assert first.indices == second.indices
    assert first.values == second.values


def test_different_texts_usually_produce_different_vectors():
    vectorizer = HashedSparseVectorizer()
    a = vectorizer.vectorize("отпуск предоставляется ежегодно")
    b = vectorizer.vectorize("зарплата выплачивается дважды в месяц")

    assert a.indices != b.indices


def test_small_vocab_size_still_stays_within_bounds():
    vectorizer = HashedSparseVectorizer(vocab_size=16)
    vector = vectorizer.vectorize("один два три четыре пять шесть семь восемь девять")

    assert all(0 <= i < 16 for i in vector.indices)

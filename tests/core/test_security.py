import time

import pytest

from app.core.security import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_verifies_against_its_own_hash():
    stored = hash_password("правильный-пароль")

    assert verify_password("правильный-пароль", stored) is True


def test_wrong_password_is_rejected():
    stored = hash_password("правильный-пароль")

    assert verify_password("неправильный", stored) is False


def test_same_password_hashes_differently_each_time():
    """Случайная соль: одинаковые пароли не должны давать одинаковые хеши."""
    first = hash_password("одинаковый")
    second = hash_password("одинаковый")

    assert first != second
    assert verify_password("одинаковый", first)
    assert verify_password("одинаковый", second)


def test_hash_does_not_contain_the_password():
    stored = hash_password("секрет123")

    assert "секрет123" not in stored


@pytest.mark.parametrize(
    "stored", ["", "мусор", "md5$1$aa$bb", "pbkdf2_sha256$нечисло$aa$bb"]
)
def test_malformed_stored_hash_is_rejected_without_raising(stored):
    try:
        assert verify_password("пароль", stored) is False
    except ValueError:
        pytest.fail("некорректный хеш не должен приводить к исключению")


def test_token_roundtrip_returns_the_subject():
    token = create_access_token("user-42")

    assert decode_access_token(token) == "user-42"


def test_tampered_payload_is_rejected():
    token = create_access_token("user-42")
    header, payload, signature = token.split(".")
    forged = create_access_token("другой-пользователь").split(".")[1]

    with pytest.raises(AuthError, match="Подпись"):
        decode_access_token(f"{header}.{forged}.{signature}")


def test_token_signed_with_another_secret_is_rejected(monkeypatch):
    from app.core.config import get_settings

    token = create_access_token("user-42")
    monkeypatch.setenv("SECRET_KEY", "совершенно-другой-секрет")
    get_settings.cache_clear()

    with pytest.raises(AuthError):
        decode_access_token(token)

    get_settings.cache_clear()


def test_expired_token_is_rejected():
    token = create_access_token("user-42", expires_in_s=1)
    time.sleep(1.1)

    with pytest.raises(AuthError, match="истёк"):
        decode_access_token(token)


@pytest.mark.parametrize("token", ["", "не-токен", "a.b", "a.b.c.d"])
def test_malformed_token_is_rejected(token):
    with pytest.raises(AuthError):
        decode_access_token(token)

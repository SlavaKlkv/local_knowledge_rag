"""Аутентификация: хеширование паролей и JWT-токены.

Секрет берётся из окружения и обязан быть переопределён в проде —
дефолтное значение годится только для локальной разработки.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

from app.core.config import get_settings
from app.core.errors import AppError

_PBKDF2_ROUNDS = 240_000
_ALGORITHM = "HS256"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 со случайной солью.

    Соль хранится вместе с хешем: без неё одинаковые пароли давали бы
    одинаковые хеши, и утечка таблицы сразу выдавала бы совпадения.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        # Битая запись в БД должна давать честный отказ в аутентификации,
        # а не 500 на попытке входа.
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
    except ValueError:
        return False
    # Сравнение с защитой от тайминг-атак.
    return hmac.compare_digest(digest.hex(), digest_hex)


def _b64encode(raw: bytes) -> str:
    return urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return urlsafe_b64decode(value + padding)


def create_access_token(subject: str, expires_in_s: int | None = None) -> str:
    settings = get_settings()
    expires_in = expires_in_s or settings.access_token_ttl_s
    issued_at = int(time.time())
    header = {"alg": _ALGORITHM, "typ": "JWT"}
    payload = {"sub": subject, "iat": issued_at, "exp": issued_at + expires_in}

    signing_input = f"{_b64encode(_dumps(header))}.{_b64encode(_dumps(payload))}"
    signature = _sign(signing_input, settings.secret_key)
    return f"{signing_input}.{signature}"


def decode_access_token(token: str) -> str:
    """Возвращает subject токена либо поднимает AuthError."""
    settings = get_settings()
    try:
        header_b64, payload_b64, signature = token.split(".")
    except ValueError as exc:
        raise AuthError("Некорректный формат токена") from exc

    expected = _sign(f"{header_b64}.{payload_b64}", settings.secret_key)
    if not hmac.compare_digest(expected, signature):
        raise AuthError("Подпись токена не совпадает")

    try:
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("Не удалось разобрать полезную нагрузку токена") from exc

    if int(payload.get("exp", 0)) <= int(time.time()):
        raise AuthError("Срок действия токена истёк")

    subject = payload.get("sub")
    if not subject:
        raise AuthError("В токене нет субъекта")
    return str(subject)


def _sign(signing_input: str, secret: str) -> str:
    signature = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return _b64encode(signature)


def _dumps(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()

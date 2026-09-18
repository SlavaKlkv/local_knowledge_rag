"""Наполняет свежую установку демо-данными.

Клон репозитория поднимается с пустой базой, и проверить платформу можно
только пройдя вручную регистрацию, создание базы знаний и загрузку файлов.
Скрипт делает это за один запуск — теми же публичными эндпоинтами, что и
веб-интерфейс, без прямого доступа к БД.

Баз знаний создаётся две, и это не украшение: разделение документов по базам
и есть способ ограничить поиск нужной областью, поэтому демо показывает его
сразу, а не оставляет платформу выглядеть однобазовой.

    uv run python -m scripts.seed_demo

Повторный запуск безопасен: существующий пользователь и одноимённая база
знаний переиспользуются, уже загруженные документы не дублируются.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-password"
DEMO_DIR = Path(__file__).resolve().parent.parent / "docs" / "demo"


@dataclass(frozen=True)
class DemoBase:
    """База знаний демо-набора и папка, из которой берутся её документы."""

    name: str
    description: str
    directory: Path


DEMO_BASES = (
    DemoBase(
        name="Внутренние документы",
        description="Демо-набор: регламент дежурств, командировки, онбординг",
        directory=DEMO_DIR / "internal",
    ),
    DemoBase(
        name="Безопасность и доступы",
        description="Демо-набор: парольная политика и доступ к продовым системам",
        directory=DEMO_DIR / "security",
    ),
)

# Индексация идёт в воркере: ждём её завершения, иначе первый же вопрос
# уйдёт по пустой базе и вернётся честным отказом.
INDEXING_TIMEOUT_S = 300
POLL_INTERVAL_S = 2.0


def _login(client: httpx.Client, email: str, password: str) -> str:
    credentials = {"email": email, "password": password}
    registered = client.post("/auth/register", json=credentials)
    if registered.status_code == 201:
        print(f"Пользователь {email} создан")
    else:
        print(f"Пользователь {email} уже существует")

    response = client.post("/auth/login", json=credentials)
    response.raise_for_status()
    return response.json()["access_token"]


def _knowledge_base(client: httpx.Client, base: DemoBase) -> str:
    existing = client.get("/knowledge-bases")
    existing.raise_for_status()
    for kb in existing.json():
        if kb["name"] == base.name:
            print(f'База знаний "{base.name}" уже есть')
            return kb["id"]

    created = client.post(
        "/knowledge-bases",
        json={"name": base.name, "description": base.description},
    )
    created.raise_for_status()
    print(f'База знаний "{base.name}" создана')
    return created.json()["id"]


def _upload(client: httpx.Client, kb_id: str, directory: Path) -> int:
    listed = client.get("/documents", params={"knowledge_base_id": kb_id})
    listed.raise_for_status()
    known = {document["filename"] for document in listed.json()}

    uploaded = 0
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        if path.name in known:
            print(f"  {path.name} — уже загружен")
            continue
        with path.open("rb") as handle:
            response = client.post(
                "/documents",
                params={"knowledge_base_id": kb_id},
                files={"file": (path.name, handle)},
            )
        response.raise_for_status()
        print(f"  {path.name} — принят")
        uploaded += 1
    return uploaded


def _wait_for_indexing(client: httpx.Client, kb_id: str) -> bool:
    deadline = time.monotonic() + INDEXING_TIMEOUT_S
    while time.monotonic() < deadline:
        response = client.get("/documents", params={"knowledge_base_id": kb_id})
        response.raise_for_status()
        documents = response.json()
        pending = [d for d in documents if d["status"] not in ("ready", "failed")]
        failed = [d for d in documents if d["status"] == "failed"]

        if not pending:
            for document in failed:
                print(f"  {document['filename']} — ошибка: {document['error']}")
            return not failed

        print(f"  индексируется: {', '.join(d['filename'] for d in pending)}")
        time.sleep(POLL_INTERVAL_S)

    print("Индексация не завершилась за отведённое время — запущен ли воркер?")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="адрес API")
    parser.add_argument("--email", default=DEMO_EMAIL)
    parser.add_argument("--password", default=DEMO_PASSWORD)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=60.0) as client:
        try:
            token = _login(client, args.email, args.password)
        except httpx.ConnectError:
            print(f"API по адресу {args.url} не отвечает — запущено ли приложение?")
            return 1

        client.headers["Authorization"] = f"Bearer {token}"

        for base in DEMO_BASES:
            kb_id = _knowledge_base(client, base)
            print("Документы:")
            if _upload(client, kb_id, base.directory):
                print("Ожидание индексации:")
                if not _wait_for_indexing(client, kb_id):
                    return 1

    print(
        "\nГотово. Откройте "
        f"{args.url} и войдите как {args.email} / {args.password}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

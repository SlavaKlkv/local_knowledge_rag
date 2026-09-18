"""Записывает демо-ролик `docs/assets/demo.mp4` по сценарию из
`docs/demo-recording-pipeline.md`.

Запись ведётся браузером по настоящим событиям мыши и клавиатуры: скрипт не
монтирует ничего поверх готового видео и не подставляет результаты — в кадр
попадает то, что интерфейс действительно показал в ответ на действие.

Курсора операционной системы в кадре нет (пишется страница, а не экран),
поэтому указатель рисуется слоем внутри страницы. Слой не имитирует
движение: он повторяет координаты настоящего события `mousemove`, того же,
что вызывает наведение и клик. Формы указателя — покадровые вырезки с
экрана macOS, они лежат в `docs/assets/cursor-shapes` и в репозиторий не
попадают.

Скрипт ждёт изолированный стенд, а не рабочий стек проекта: он создаёт и
удаляет базы знаний, грузит документы и переписывает состав кольца.

    uv run python -m scripts.record_demo --url http://localhost:8010
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CURSOR_DIR = ROOT / "docs" / "assets" / "cursor-shapes"
DEMO_DIR = ROOT / "docs" / "demo"
OUTPUT = ROOT / "docs" / "assets" / "demo.mp4"

DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-password"

INTERNAL_KB = "Внутренние документы"
SECURITY_KB = "Безопасность и доступы"

# Кадр 1180x800 в CSS-пикселях, а масштаб 125% задаётся зумом страницы —
# ровно тем же, что даёт браузерный зум. Апскейла при этом нет: всё, что
# видно в кадре, отрисовано в его собственном разрешении.
ZOOM = 1.25
VIEWPORT = {"width": 1180, "height": 800}
VIDEO_SIZE = {"width": 1180, "height": 800}

CHAT_SUGGESTION = "Сколько длится испытательный срок?"
SEARCH_QUERY = "в какие сроки нужно уложиться"

# Размер указателя задаётся не «натуральной» величиной вырезки, а его долей
# от интерфейса на настоящей записи экрана: кадр демо — тесная вырезка
# 1180x800 при зуме 125%, и те же кнопки занимают в нём заметно большую часть
# площади, чем на реальном экране. Курсор, нарисованный в натуральных
# CSS-пикселях, выходил в кадре вдвое крупнее, чем на записи экрана рядом с
# тем же интерфейсом. По контрольным кадрам карточка входа одинаковой ширины
# (381 против 380 px), а стрелка — 8x13 против 18x25; отсюда и множитель.
CURSOR_SCALE = 0.26

# Горячая точка каждой формы в долях от её размера: стрелка и рука
# указывают остриём, текстовая черта и ладонь — серединой.
CURSOR_SHAPES = {
    "default": ("default.png", 0.12, 0.08),
    "pointer": ("pointer.png", 0.35, 0.10),
    "text": ("text.png", 0.50, 0.50),
    "grabbing": ("grab.png", 0.50, 0.50),
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def _cursor_assets() -> dict[str, dict[str, object]]:
    assets: dict[str, dict[str, object]] = {}
    for name, (filename, hot_x, hot_y) in CURSOR_SHAPES.items():
        path = CURSOR_DIR / filename
        if not path.exists():
            raise SystemExit(
                f"Нет формы курсора {path}. Вырезки не хранятся в репозитории — "
                "их нужно нарезать с записи экрана перед записью демо."
            )
        data = base64.b64encode(path.read_bytes()).decode()
        assets[name] = {
            "src": f"data:image/png;base64,{data}",
            "hotX": hot_x,
            "hotY": hot_y,
        }
    return assets


CURSOR_LAYER_JS = """
(assets) => {
  const SCALE = __SCALE__;
  const layer = document.createElement('div');
  layer.id = '__demo_cursor';
  Object.assign(layer.style, {
    position: 'fixed',
    left: '0px',
    top: '0px',
    zIndex: '2147483647',
    pointerEvents: 'none',
    willChange: 'transform',
    transform: 'translate(-1000px, -1000px)',
  });

  const images = {};
  for (const [name, info] of Object.entries(assets)) {
    const img = new Image();
    img.src = info.src;
    Object.assign(img.style, {
      position: 'absolute',
      display: 'none',
      imageRendering: 'auto',
    });
    img.onload = () => {
      // Вырезка снята в двойном разрешении: в CSS-пикселях она вдвое меньше.
      const w = img.naturalWidth * SCALE;
      const h = img.naturalHeight * SCALE;
      img.style.width = w + 'px';
      img.style.height = h + 'px';
      img.style.left = -(w * info.hotX) + 'px';
      img.style.top = -(h * info.hotY) + 'px';
    };
    images[name] = img;
    layer.append(img);
  }

  const show = (name) => {
    for (const [key, img] of Object.entries(images)) {
      img.style.display = key === name ? 'block' : 'none';
    }
  };
  show('default');

  // Зум висит на body, поэтому слой живёт в documentElement: иначе
  // указатель растянулся бы вместе со страницей и разошёлся с координатами
  // настоящих событий мыши.
  // Init-скрипт выполняется до разбора документа, поэтому слой цепляется
  // при первой же возможности и переподключается, если приложение
  // перерисовало корень.
  const mount = () => {
    const root = document.documentElement;
    if (root && !layer.isConnected) root.append(layer);
  };
  mount();
  document.addEventListener('DOMContentLoaded', mount);

  window.__demoCursor = {
    shape: 'default',
    dragging: false,
    move(x, y) {
      mount();
      layer.style.transform = `translate(${x}px, ${y}px)`;
      if (this.dragging) return;
      // Форма берётся из настоящего вычисленного стиля элемента под
      // курсором — той же, что показала бы система.
      const el = document.elementFromPoint(x, y);
      let shape = 'default';
      if (el) {
        const css = getComputedStyle(el).cursor;
        if (css === 'pointer') shape = 'pointer';
        else if (css === 'text') shape = 'text';
        else if (css === 'grab' || css === 'grabbing') shape = 'grabbing';
        else if (el.closest('.draggable-row')) shape = 'grabbing';
      }
      this.shape = shape;
      show(shape);
    },
    // Пока строку несут, система показывает стрелку — так и здесь.
    setDragging(on) {
      this.dragging = on;
      if (on) show('default');
    },
  };

  document.addEventListener(
    'mousemove',
    (event) => window.__demoCursor.move(event.clientX, event.clientY),
    true,
  );

  // Пока идёт настоящее перетаскивание, браузер перестаёт слать `mousemove`:
  // указатель замирал на месте, хотя строка ехала. Координаты на это время
  // берутся из событий самого драга — они такие же настоящие и приходят с той
  // же частотой.
  for (const name of ['drag', 'dragover']) {
    document.addEventListener(
      name,
      (event) => {
        if (!event.clientX && !event.clientY) return;
        window.__demoCursor.move(event.clientX, event.clientY);
      },
      true,
    );
  }
}
""".replace("__SCALE__", str(CURSOR_SCALE))  # f-строка тут не годится: в JS свои {}


class Recorder:
    """Ведёт курсор и клавиатуру так, как это делает человек."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.pos = Point(VIEWPORT["width"] * 0.5, VIEWPORT["height"] * 0.5)

    # --- Движение ---

    def move_to(self, x: float, y: float, duration: float = 0.26) -> None:
        """Подводит курсор с замедлением у цели.

        Равномерное движение сразу выдаёт автомат, поэтому скорость падает
        по мере приближения (ease-out), а шаг привязан к кадру записи.
        """
        start = self.pos
        steps = max(2, int(duration / 0.016))
        for i in range(1, steps + 1):
            t = i / steps
            eased = 1 - math.pow(1 - t, 3)
            nx = start.x + (x - start.x) * eased
            ny = start.y + (y - start.y) * eased
            self.page.mouse.move(nx, ny)
            self.page.wait_for_timeout(16)
        self.pos = Point(x, y)

    def hover(self, selector: str, duration: float = 0.26, settle: float = 0.12):
        """Наводит курсор на середину элемента и даёт увидеть наведение."""
        element = self.page.locator(selector).first
        element.wait_for(state="visible")
        element.scroll_into_view_if_needed()
        box = element.bounding_box()
        if box is None:
            raise RuntimeError(f"Элемент {selector} не виден на странице")
        self.move_to(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, duration)
        self.pause(settle)
        return element

    def click(self, selector: str, duration: float = 0.26, settle: float = 0.12) -> None:
        self.hover(selector, duration, settle)
        self.page.mouse.down()
        self.page.wait_for_timeout(70)
        self.page.mouse.up()

    def type(self, selector: str, text: str, delay: float = 40) -> None:
        """Набирает текст посимвольно — мгновенная вставка в кадре видна."""
        self.click(selector, settle=0.18)
        self.page.wait_for_timeout(140)
        self.page.keyboard.type(text, delay=delay)

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def scroll_to_bottom(self, selector: str, steps: int = 18) -> None:
        """Прокручивает контейнер колесом ровно до низа и там останавливается.

        Крутить «с запасом» нельзя: обороты колеса, пришедшие на упор,
        Chromium отыгрывает отдачей, и содержимое несколько кадров дрожит на
        пару пикселей — в кадре это выглядит как подёргивание после скролла.
        Поэтому дистанция берётся по факту: перед каждым шагом спрашиваем,
        сколько осталось, и последним шагом проходим ровно остаток.
        """
        remaining = self._scroll_left(selector)
        if remaining <= 1:
            return
        # Зум страницы разводит пиксели колеса и пиксели прокрутки, поэтому
        # цена одного оборота не угадывается, а измеряется на первом шаге.
        wheel_per_px = 1.0
        chunk = remaining / steps
        for _ in range(steps):
            left = self._scroll_left(selector)
            if left <= 1:
                break
            before = left
            self.page.mouse.wheel(0, min(chunk, left) * wheel_per_px)
            # Шаг привязан к кадру записи: на более редких оборотах колеса
            # прокрутка в кадре идёт заметными скачками.
            self.page.wait_for_timeout(32)
            moved = before - self._scroll_left(selector)
            if moved > 1:
                wheel_per_px = min(chunk, before) * wheel_per_px / moved

    def _scroll_left(self, selector: str) -> float:
        return self.page.evaluate(
            "(sel) => { const el = document.querySelector(sel);"
            " return el ? el.scrollHeight - el.clientHeight - el.scrollTop : 0; }",
            selector,
        )

    # --- Перетаскивание ---

    def drag(self, source: str, target: str, duration: float = 1.2) -> None:
        """Переносит строку кольца одним непрерывным движением.

        Браузер сам несёт за курсором копию карточки и подсвечивает строку
        под ней: событий хватает настоящих, дорисовывать нечего.
        """
        box = self.hover(source, duration=0.4, settle=0.25).bounding_box()
        assert box is not None
        self.page.evaluate("() => window.__demoCursor.setDragging(true)")
        self.page.mouse.down()
        self.page.wait_for_timeout(120)

        destination = self.page.locator(target).first
        end = destination.bounding_box()
        assert end is not None
        # Целимся в нижнюю половину строки: вставка идёт на её место.
        ex = end["x"] + end["width"] / 2
        ey = end["y"] + end["height"] * 0.6

        start = self.pos
        steps = max(2, int(duration / 0.016))
        for i in range(1, steps + 1):
            t = i / steps
            eased = 1 - math.pow(1 - t, 3)
            nx = start.x + (ex - start.x) * eased
            ny = start.y + (ey - start.y) * eased
            self.page.mouse.move(nx, ny)
            # Событий драга хватает, чтобы вести указатель, но приходят они
            # реже шагов движения. Слой ставится по координатам того самого
            # события мыши, которое только что отправлено, — иначе указатель
            # отстаёт от строки рывками.
            self.page.evaluate(
                "([x, y]) => window.__demoCursor.move(x, y)", [nx, ny]
            )
            self.page.wait_for_timeout(16)
        self.pos = Point(ex, ey)

        # Карточку придерживают над целью, прежде чем отпустить.
        self.pause(0.4)
        self.page.mouse.up()
        self.page.evaluate("() => window.__demoCursor.setDragging(false)")


def reset_stand(url: str) -> None:
    """Возвращает стенд в начальное состояние сценария.

    У аккаунта остаётся одна база `Внутренние документы` с тремя
    проиндексированными документами, состав кольца не переопределён.
    """
    with httpx.Client(base_url=url, timeout=120.0) as client:
        credentials = {"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
        client.post("/auth/register", json=credentials)
        token = client.post("/auth/login", json=credentials).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"

        bases = client.get("/knowledge-bases").json()
        for base in bases:
            if base["name"] == SECURITY_KB:
                client.delete(f"/knowledge-bases/{base['id']}")
                print(f'База "{SECURITY_KB}" удалена')

        internal = next((b for b in bases if b["name"] == INTERNAL_KB), None)
        if internal is None:
            created = client.post(
                "/knowledge-bases",
                json={
                    "name": INTERNAL_KB,
                    "description": "Демо-набор: регламент дежурств, командировки, онбординг",
                },
            )
            created.raise_for_status()
            internal = created.json()
            for path in sorted((DEMO_DIR / "internal").iterdir()):
                with path.open("rb") as handle:
                    client.post(
                        "/documents",
                        params={"knowledge_base_id": internal["id"]},
                        files={"file": (path.name, handle)},
                    )
            _wait_indexed(client, internal["id"])

        client.delete("/inference/ring")
        print("Состав кольца сброшен к профилю")


def _wait_indexed(client: httpx.Client, kb_id: str) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        documents = client.get("/documents", params={"knowledge_base_id": kb_id}).json()
        if all(d["status"] in ("ready", "failed") for d in documents):
            return
        time.sleep(2)
    raise SystemExit("Индексация не завершилась — запущен ли воркер?")


def scenario(page: Page, url: str) -> None:
    rec = Recorder(page)
    started = time.monotonic()

    def step(name: str) -> None:
        # Ролик ограничен сорока секундами, и режется он не на глаз:
        # видно, какой шаг сколько занял в этом прогоне.
        print(f"  {time.monotonic() - started:6.1f}s  {name}")

    # --- Начальное состояние: страница входа, тёмная тема, курсор слева
    # от карточки и до первого действия неподвижен.
    step("goto")
    page.goto(f"{url}/")
    page.wait_for_selector("#auth-form", state="visible")
    card = page.locator(".auth-card").bounding_box()
    assert card is not None
    rec.pos = Point(card["x"] - 60, card["y"] + card["height"] * 0.62)
    page.mouse.move(rec.pos.x, rec.pos.y)
    rec.pause(0.6)

    # 1. Вход — с реальным наведением на кнопку.
    step("login")
    rec.click("#auth-form button.primary")
    page.wait_for_selector("#main-view", state="visible")
    page.wait_for_selector("#kb-list li", state="visible")
    rec.pause(0.4)

    # 2-3. Тёмная → светлая → системная.
    step("themes")
    rec.click("#theme-switch button[data-theme='light']")
    rec.pause(0.45)
    rec.click("#theme-switch button[data-theme='system']")
    rec.pause(0.5)

    # 4. Новая база знаний — название набирается с клавиатуры.
    step("new kb")
    rec.click("#kb-new")
    rec.pause(0.2)
    rec.type("#kb-name", SECURITY_KB)
    rec.pause(0.15)
    rec.click("#kb-form button[type='submit']")
    page.wait_for_selector(f"#kb-list li:has-text('{SECURITY_KB}')")
    rec.pause(0.4)

    # 5-8. Два документа через настоящий файловый input.
    step("uploads")
    for filename in ("parolnaya-politika.md", "dostup-k-prod.txt"):
        rec.hover(".upload", settle=0.25)
        with page.expect_file_chooser() as chooser:
            page.mouse.down()
            page.wait_for_timeout(70)
            page.mouse.up()
        chooser.value.set_files(DEMO_DIR / "security" / filename)
        page.wait_for_selector(
            f"#doc-list li:has-text('{filename}') .status-indexed",
            timeout=180_000,
        )
        rec.pause(0.3)

    # 9. Вопрос задаётся подсказкой: набор уже показан на создании базы.
    step("chat")
    rec.click("#suggestions button:has-text('Какой минимальной')", settle=0.25)
    page.wait_for_selector(".msg-bot .answer, .msg-bot .sources", timeout=180_000)
    rec.pause(0.7)

    # 10. Переключение на базу с внутренними документами.
    step("switch kb")
    rec.click(f"#kb-list li:has-text('{INTERNAL_KB}')")
    rec.pause(0.5)

    # 11-13. Поиск без генерации: один запрос — несколько фрагментов из
    # разных документов, по убыванию релевантности.
    step("search")
    rec.click(".tab[data-tab='search']")
    rec.pause(0.2)
    rec.type("#search-input", SEARCH_QUERY)
    rec.pause(0.15)
    rec.click("#search-form button.primary")
    page.wait_for_selector("#search-results .hit", timeout=120_000)
    rec.pause(0.9)

    # 14-15. Настройка кольца: показать наведение, затем прокрутить до низа.
    step("ring open")
    rec.click("#ring-setup")
    page.wait_for_selector("#ring-config .ring-composition")
    rec.pause(0.4)
    rec.move_to(VIEWPORT["width"] * 0.55, VIEWPORT["height"] * 0.6, duration=0.35)
    rec.scroll_to_bottom("#ring-config")
    rec.pause(0.4)

    # 16. Модель вне профиля железа выбирается из списка кандидатов.
    step("candidate")
    rec.click(".picker-trigger")
    rec.pause(0.2)
    rec.click(".picker-option:has-text('llava:latest')", settle=0.35)
    rec.pause(0.2)
    rec.click(".ring-composition button:has-text('Добавить')")
    rec.pause(0.45)

    # 17. Перенос основной модели в конец кольца.
    step("drag")
    rows = ".ring-composition .draggable-row"
    rec.drag(f"{rows}:has-text('qwen3:4b')", f"{rows}:nth-child(3)")
    rec.pause(0.5)

    # 18-19. Сохранение и плавное исчезновение подтверждения.
    step("save")
    rec.click("button:has-text('Сохранить состав')")
    page.wait_for_selector("#ring-message", state="visible")
    # Подтверждение уходит само через пять секунд — это и показывается,
    # кликать по нему не нужно.
    page.wait_for_selector("#ring-message", state="hidden", timeout=15_000)
    rec.pause(0.4)

    # 20. Закрыть окно и показать новый порядок кольца в боковой панели.
    step("close")
    rec.click("#ring-close")
    rec.pause(1.0)


def convert(webm: Path, output: Path) -> None:
    """Перекодирует запись в H.264 для README.

    Начало записи в ролик не попадает: первый кадр — страница до применения
    темы, он белый, а на следующем тема уже тёмная, но указателя ещё нет —
    слой ставится первым движением мыши. Первым кадром ролика должно быть
    рабочее состояние целиком, вместе с курсором, поэтому отбрасываются оба.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(webm),
            "-vf",
            "select='gte(n\\,2)',setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8010", help="адрес стенда")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--keep-webm", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="запускать браузер с окном, а не headless",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("Нужен ffmpeg: запись пишется в webm и перекодируется в mp4")

    if not args.no_reset:
        reset_stand(args.url)

    assets = _cursor_assets()
    videos = args.output.parent / "_recording"
    if videos.exists():
        shutil.rmtree(videos)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            args=["--hide-scrollbars"],
        )
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(videos),
            record_video_size=VIDEO_SIZE,
            locale="ru-RU",
            # Ролик открывается страницей входа с активной тёмной темой:
            # выбор темы живёт в localStorage, токена в хранилище нет.
            storage_state={
                "cookies": [],
                "origins": [
                    {
                        "origin": args.url,
                        "localStorage": [{"name": "lkr.theme", "value": "dark"}],
                    }
                ],
            },
        )
        # add_init_script не принимает аргументов: формы вшиваются в сам скрипт.
        context.add_init_script(f"({CURSOR_LAYER_JS})({json.dumps(assets)})")
        # Масштаб 125% — зумом страницы: тем же, что даёт браузерный зум.
        context.add_init_script(
            "() => {"
            "  const apply = () => {"
            f"    if (document.body) document.body.style.zoom = '{ZOOM}';"
            "  };"
            "  apply();"
            "  document.addEventListener('DOMContentLoaded', apply);"
            "}"
        )
        page = context.new_page()
        try:
            scenario(page, args.url)
        finally:
            video = page.video
            context.close()
            browser.close()
            if video is not None:
                source = Path(video.path())
                convert(source, args.output)
                if not args.keep_webm:
                    shutil.rmtree(videos, ignore_errors=True)
                print(f"Готово: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

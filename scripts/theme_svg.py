"""Пересобирает светлую версию SVG-схемы из тёмной.

Обе версии подключены в README через <picture> с prefers-color-scheme, и
расходятся они молча: правку вносят в одну, а читатель со второй темой видит
старую схему. Поэтому светлый вариант не редактируется руками — правится
только `*-dark.svg`, а затем запускается:

    python scripts/theme_svg.py

Скрипт заменяет цвета по карте ниже; любой новый цвет в тёмной версии, для
которого нет пары, обрывает сборку с ошибкой, а не переносится как есть.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

# тёмный цвет -> светлый
PALETTE = {
    "#000000": "#000000",  # тень
    "#08080d": "#ffffff",  # фон полотна (в светлой версии переопределяется ниже)
    "#0d1117": "#ffffff",  # фон полотна GitHub Light
    "#101018": "#ffffff",  # заливка карточек
    "#e8eaf2": "#1b2030",  # основной текст
    "#e2e3e8": "#ffffff",  # текст на акцентном бейдже
    "#9aa3b8": "#5b657a",  # пояснения
    "#cfd6e6": "#596579",  # нейтральные стрелки
    "#ec4899": "#b84e78",  # Qdrant
    "#c43d7d": "#a94c72",
    "#8e2c60": "#c96389",
    "#ffd0e5": "#fff5f8",
    "#fff1f7": "#ffffff",
    "#f472b6": "#b84e78",
    "#22d3ee": "#458292",  # dense
    "#eab308": "#9a7a2f",  # sparse
    "#a78bfa": "#7560a8",  # payload, итоговый ответ
    "#60a5fa": "#4169a8",  # документ, PostgreSQL
    "#9c78d3": "#7560a8",  # связь чанка с точкой, переиндексация
    "#34d399": "#047857",  # вопрос
    "#2fb88a": "#43856f",  # payload
    "#af5800": "#a66a3d",  # векторы
    "#afaf00": "#9a7a2f",
    "#af0000": "#b65b5b",
    "#160910": "#fdf2f7",
    "#0d0a10": "#fdf2f7",
    "#150a11": "#fff5f9",
    "#0a1417": "#ecfeff",
    "#0e1b20": "#e0f7fb",
    "#1a1608": "#fefce8",
    "#0f0a18": "#f5f3ff",
    "#0b0b12": "#fdfdff",  # подложка секции
    "#140a11": "#fdf2f7",
    "#232838": "#d3d8e6",  # рамка секции и разделители
    "#0d1d33": "#d6e7fa",
    "#14283d": "#c9dcf2",
    "#121722": "#e1ecf8",
    "#2a0b1b": "#f5dbe6",
    "#08262e": "#d9edf1",
    "#292106": "#f5e8bd",
    "#1b1429": "#ded0ef",
    "#1c0f19": "#f5dbe6",
    "#292930": "#f8fafc",
    "#383842": "#e5e7eb",
    "#99a2b7": "#475569",
    "#e7e9f1": "#94a3b8",
    "#24170c": "#f7e4d2",
    "#3b3b08": "#f5e7ad",
    "#3b0a0a": "#f6cccc",
    "#0b231c": "#d7eee4",
    "#0c1d19": "#e1f2ea",
}

FILE_PALETTES = {
    "how-it-works-dark.svg": {
        "#cfd6e6": "#475569",  # стрелки
        "#60a5fa": "#2563eb",  # файл
        "#0d1d33": "#bfdbfe",
        "#2fb88a": "#16a34a",  # успешный ответ
        "#0b231c": "#bbf7d0",
        "#232838": "#cbd5e1",  # очередь, RRF, reranker
        "#9aa3b8": "#64748b",
        "#9c78d3": "#7c3aed",  # воркер
        "#1b1429": "#ddd6fe",
        "#ec4899": "#db2777",  # индекс и фрагменты
        "#2a0b1b": "#fbcfe8",
        "#af5800": "#ea580c",  # поиск
        "#24170c": "#fed7aa",
        "#383842": "#e2e8f0",  # вопрос и LLM
    },
}

LIGHT_CANVAS = ".bg   { fill: #ffffff; }"


def convert(dark: Path, light: Path) -> None:
    source = dark.read_text(encoding="utf-8")
    palette = PALETTE | FILE_PALETTES.get(dark.name, {})
    unknown = {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}", source)} - palette.keys()
    if unknown:
        raise SystemExit(
            f"{dark.name}: нет светлой пары для цветов {sorted(unknown)} — "
            f"добавьте их в PALETTE в {Path(__file__).name}"
        )
    result = re.sub(
        r"#[0-9a-fA-F]{6}",
        lambda match: palette[match.group(0).lower()],
        source,
    )
    result = result.replace(".bg   { fill: #ffffff; }", LIGHT_CANVAS)
    result = result.replace('flood-opacity="0.32"', 'flood-opacity="0.16"')
    light.write_text(result, encoding="utf-8")
    print(f"{dark.name} -> {light.name}")


def main() -> int:
    dark_files = sorted(ASSETS.glob("*-dark.svg"))
    if not dark_files:
        raise SystemExit(f"в {ASSETS} нет ни одного *-dark.svg")
    for dark in dark_files:
        convert(dark, dark.with_name(dark.name.replace("-dark.svg", "-light.svg")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

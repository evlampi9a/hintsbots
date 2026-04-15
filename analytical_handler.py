"""
Analytical handler for Hints bot.
Handles statistical queries about the KP database: median prices, averages, distributions.
"""
import requests
import os
import logging
from telegram import Update
from telegram.ext import ContextTypes
from openai import OpenAI

logger = logging.getLogger(__name__)

SB_URL = os.environ["SB_URL"]
SB_KEY = os.environ["SB_KEY"]
SB_H = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Content-Type": "application/json"}

OPENAI_KEY = os.environ["OPENAI_KEY"]
oai = OpenAI(api_key=OPENAI_KEY, base_url="https://api.openai.com/v1")

ANALYTICAL_KEYWORDS = [
    "медиан", "средн", "статистика", "аналитика",
    "сколько в среднем", "какая медианная", "какая средняя",
    "средняя стоимость", "медианная цена", "средняя цена",
    "сколько стоит в среднем", "диапазон цен", "распределение цен",
    "сколько проектов", "как часто", "популярные работы",
    "топ работ", "топ сфер", "какие работы чаще"
]

# Маппинг сфер для нормализации
SPHERE_ALIASES = {
    "финтех": "FinTech",
    "fintech": "FinTech",
    "финансы": "FinTech",
    "эдтех": "EdTech",
    "edtech": "EdTech",
    "образование": "EdTech",
    "ит": "IT",
    "it": "IT",
    "айти": "IT",
    "хртех": "HRTech",
    "hrtech": "HRTech",
    "hr": "HRTech",
    "ритейл": "Ритейл",
    "retail": "Ритейл",
    "стартап": "Стартапы и поиск идеи для продукта",
}

# Маппинг стран
COUNTRY_ALIASES = {
    "россия": "Россия",
    "рф": "Россия",
    "кения": "Кения",
    "оаэ": "ОАЭ",
    "сша": "США",
    "usa": "США",
    "индия": "Индия",
    "китай": "Китай",
}

def is_analytical_query(text: str) -> bool:
    """Определяет, является ли запрос аналитическим."""
    t = text.lower()
    return any(keyword in t for keyword in ANALYTICAL_KEYWORDS)


def fetch_all_projects() -> list[dict]:
    """Загружает все проекты из kp_projects."""
    all_rows = []
    offset = 0
    limit = 1000
    while True:
        r = requests.get(
            f"{SB_URL}/rest/v1/kp_projects",
            headers={k: v for k, v in SB_H.items() if k != "Content-Type"},
            params={
                "select": "project_name,company,sphere,industry,client_price,project_type,works_tags,countries,interview_depth,b2b_segment",
                "limit": limit,
                "offset": offset,
            },
            timeout=20,
        )
        if r.status_code != 200:
            break
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
    return all_rows


def parse_filters(text: str) -> dict:
    """Извлекает фильтры из текста запроса."""
    t = text.lower()
    filters = {}

    # Сфера
    for alias, sphere in SPHERE_ALIASES.items():
        if alias in t:
            filters["sphere"] = sphere
            break

    # Страна
    for alias, country in COUNTRY_ALIASES.items():
        if alias in t:
            filters["country"] = country
            break

    # B2B/B2C
    if "b2b" in t:
        filters["segment"] = "B2B"
    elif "b2c" in t:
        filters["segment"] = "B2C"

    return filters


def compute_stats(projects: list[dict], filters: dict) -> dict:
    """Вычисляет статистику по проектам с учётом фильтров."""
    # Применяем фильтры
    filtered = projects
    if "sphere" in filters:
        filtered = [p for p in filtered if (p.get("sphere") or "").lower() == filters["sphere"].lower()
                    or (p.get("industry") or "").lower() == filters["sphere"].lower()]
    if "country" in filters:
        target = filters["country"].lower()
        filtered = [p for p in filtered if
                    any(target in (c or "").lower() for c in (p.get("countries") or []))
                    or target in (p.get("country") or "").lower()]
    if "segment" in filters:
        seg = filters["segment"]
        filtered = [p for p in filtered if seg.lower() in (p.get("b2b_segment") or "").lower()
                    or seg.lower() in (p.get("interview_depth") or "").lower()]

    prices = [p["client_price"] for p in filtered if p.get("client_price") and p["client_price"] > 0]
    prices.sort()

    stats = {
        "total": len(filtered),
        "with_price": len(prices),
    }

    if prices:
        stats["min"] = prices[0]
        stats["max"] = prices[-1]
        stats["median"] = prices[len(prices) // 2]
        stats["mean"] = int(sum(prices) / len(prices))
        # Квартили
        q1_idx = len(prices) // 4
        q3_idx = 3 * len(prices) // 4
        stats["q1"] = prices[q1_idx]
        stats["q3"] = prices[q3_idx]

    # Топ сфер
    sphere_counts = {}
    for p in filtered:
        s = p.get("sphere") or p.get("industry") or "Не указана"
        sphere_counts[s] = sphere_counts.get(s, 0) + 1
    stats["top_spheres"] = sorted(sphere_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Топ типов работ
    works_counts = {}
    for p in filtered:
        tags = p.get("works_tags") or []
        if isinstance(tags, list):
            for tag in tags:
                if tag:
                    works_counts[tag] = works_counts.get(tag, 0) + 1
    stats["top_works"] = sorted(works_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return stats


def format_price(p: int) -> str:
    """Форматирует цену в читаемый вид."""
    if p >= 1_000_000:
        return f"{p/1_000_000:.1f} млн ₽"
    elif p >= 1_000:
        return f"{p//1_000} тыс. ₽"
    return f"{p} ₽"


async def handle_analytical_query(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Обрабатывает аналитический запрос — считает статистику по базе."""
    await update.message.reply_text("📊 Считаю статистику по базе...")
    try:
        projects = fetch_all_projects()
        filters = parse_filters(text)
        stats = compute_stats(projects, filters)

        filter_desc = ""
        if filters.get("sphere"):
            filter_desc += f" в сфере <b>{filters['sphere']}</b>"
        if filters.get("country"):
            filter_desc += f" в стране <b>{filters['country']}</b>"
        if filters.get("segment"):
            filter_desc += f" (сегмент <b>{filters['segment']}</b>)"

        lines = [f"📊 <b>Статистика по базе Hints{filter_desc}</b>\n"]
        lines.append(f"Всего проектов: <b>{stats['total']}</b> (с ценой: {stats['with_price']})\n")

        if stats.get("median"):
            lines.append("<b>Цены (клиентские):</b>")
            lines.append(f"- Медиана: <b>{format_price(stats['median'])}</b>")
            lines.append(f"- Среднее: {format_price(stats['mean'])}")
            lines.append(f"- Q1–Q3: {format_price(stats['q1'])} — {format_price(stats['q3'])}")
            lines.append(f"- Диапазон: {format_price(stats['min'])} — {format_price(stats['max'])}\n")

        if stats.get("top_spheres"):
            lines.append("<b>Топ сфер:</b>")
            for sphere, cnt in stats["top_spheres"]:
                lines.append(f"- {sphere}: {cnt} проектов")
            lines.append("")

        if stats.get("top_works"):
            lines.append("<b>Топ типов работ:</b>")
            for work, cnt in stats["top_works"]:
                lines.append(f"- {work}: {cnt} проектов")

        response = "\n".join(lines)
        await update.message.reply_text(response, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Analytical error: {e}")
        await update.message.reply_text("Ошибка при подсчёте статистики. Попробуй ещё раз.")

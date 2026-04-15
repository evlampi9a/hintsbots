"""
Similar projects handler for Hints bot.
Finds and filters projects from kp_projects by parameters.
"""
import requests
import os
import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

SB_URL = os.environ["SB_URL"]
SB_KEY = os.environ["SB_KEY"]
SB_H = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Content-Type": "application/json"}

SIMILAR_KEYWORDS = [
    "найди похожие", "похожие проекты", "покажи проекты",
    "какие проекты", "проекты по", "проекты для", "проекты с",
    "найди проекты", "поиск проектов", "похожий проект",
    "аналогичные проекты", "примеры проектов", "проекты в сфере",
    "фильтр", "отфильтруй", "покажи все проекты", "проекты в ",
    "пользовательские интервью", "пользовательскими интервью",
    "глубинные интервью", "глубинными интервью", "глубинки",
    "экспертные интервью", "экспертными интервью", "экспертки",
    "тайная закупка", "опросы", "опросов", "дневники",
    "кастдев", "jtbd", "кано",
]

# Маппинг индустрий
INDUSTRY_MAP = {
    "финтех": "FinTech", "fintech": "FinTech", "финансы": "FinTech",
    "финансовый": "FinTech", "банк": "FinTech",
    "эдтех": "EdTech", "edtech": "EdTech", "образование": "EdTech",
    "образовательный": "EdTech", "обучение": "EdTech",
    "хртех": "HRTech", "hrtech": "HRTech",
    "ритейл": "Ритейл", "retail": "Ритейл", "торговля": "Ритейл",
    "магазин": "Ритейл", "маркетплейс": "Ритейл",
    "логистика": "Логистика", "доставка": "Логистика",
    "недвижимость": "Недвижимость", "застройщик": "Недвижимость",
    "девелопер": "Недвижимость", "девелоп": "Недвижимость",
    "медицина": "Медицина и здоровье", "здоровье": "Медицина и здоровье",
    "фарма": "Медицина и здоровье", "pharma": "Медицина и здоровье",
    "авто": "Авто", "автомобил": "Авто",
    "туризм": "Туризм и гостиничный бизнес", "отел": "Туризм и гостиничный бизнес",
    "гостиниц": "Туризм и гостиничный бизнес",
    "телеком": "Телеком", "telecom": "Телеком",
    "страхован": "Страхование", "страховк": "Страхование",
    "энергетик": "Энергетика", "нефт": "Нефть и газ", "газ": "Нефть и газ",
    "стартап": "Стартапы и поиск идеи для продукта",
    "fmcg": "FMCG", "продукт": "FMCG",
    "b2b": None, "b2c": None,
}

# Маппинг типов работ
WORKS_KEYWORDS = {
    "глубинные интервью": "Глубинные интервью",
    "глубинки": "Глубинные интервью",
    "пользовательские интервью": "Глубинные интервью",
    "экспертные интервью": "Экспертные интервью",
    "экспертки": "Экспертные интервью",
    "фокус-группы": "Фокус-группы",
    "фокус группы": "Фокус-группы",
    "тайная закупка": "Тайная закупка",
    "mystery shopping": "Тайная закупка",
    "опросы": "Опросы",
    "количественное": "Опросы",
    "кабинетное": "Кабинетное исследование",
    "desk research": "Кабинетное исследование",
    "дневники": "Дневниковые исследования",
    "кастдев": "CustDev",
    "custdev": "CustDev",
    "jtbd": "JTBD",
    "кано": "Кано",
    "kano": "Кано",
    "ux": "UX-исследование",
    "юзабилити": "UX-исследование",
    "рекрутинг": "Рекрутинг",
    "рекрут": "Рекрутинг",
}

# Маппинг стран
COUNTRY_ALIASES = {
    "россия": "Россия", "рф": "Россия", "российск": "Россия",
    "оаэ": "ОАЭ", "uae": "ОАЭ", "дубай": "ОАЭ", "dubai": "ОАЭ",
    "саудовск": "Саудовская Аравия", "ksa": "Саудовская Аравия", "saudi": "Саудовская Аравия",
    "катар": "Катар", "qatar": "Катар",
    "казахстан": "Казахстан", "kz": "Казахстан",
    "узбекистан": "Узбекистан",
    "беларус": "Беларусь",
    "украин": "Украина",
    "кения": "Кения", "kenya": "Кения",
    "индия": "Индия", "india": "Индия",
    "китай": "Китай", "china": "Китай",
    "сша": "США", "usa": "США", "америк": "США",
    "европ": "Европа",
    "германи": "Германия", "germany": "Германия",
    "великобритани": "Великобритания", "uk": "Великобритания",
    "франци": "Франция", "france": "Франция",
    "mena": "MENA",
    "снг": "СНГ",
}

# Маппинг компаний
COMPANY_ALIASES = {
    "яндекс": "Яндекс", "yandex": "Яндекс",
    "сбер": "Сбер", "сбербанк": "Сбер",
    "авито": "Авито", "avito": "Авито",
    "пятёрочка": "Пятёрочка", "пятерочка": "Пятёрочка",
    "самолёт": "Самолёт", "самолет": "Самолёт",
    "т-банк": "Т-Банк", "тинькофф": "Т-Банк", "tinkoff": "Т-Банк",
    "вкусвилл": "ВкусВилл",
    "мтс": "МТС",
    "билайн": "Билайн",
    "мегафон": "МегаФон",
    "озон": "Ozon", "ozon": "Ozon",
    "вб": "Wildberries", "wildberries": "Wildberries",
    "ламода": "Lamoda", "lamoda": "Lamoda",
    "ростелеком": "Ростелеком",
    "газпром": "Газпром",
    "лукойл": "Лукойл",
    "ржд": "РЖД",
    "аэрофлот": "Аэрофлот",
    "втб": "ВТБ",
    "альфа": "Альфа-Банк",
    "сфера": "Сфера",
}


def is_similar_query(text: str) -> bool:
    """Определяет, является ли запрос поиском похожих проектов."""
    t = text.lower()
    if any(kw in t for kw in SIMILAR_KEYWORDS):
        return True
    return False


def parse_similar_filters(text: str) -> dict:
    """Извлекает фильтры для поиска похожих проектов."""
    t = text.lower()
    filters = {}

    # Индустрия/сфера
    for alias, industry in INDUSTRY_MAP.items():
        if alias in t and industry:
            filters["industry"] = industry
            break

    # B2B/B2C сегмент
    if "b2b" in t:
        filters["segment"] = "B2B"
    elif "b2c" in t:
        filters["segment"] = "B2C"

    # Страна
    for alias, country in COUNTRY_ALIASES.items():
        if alias in t:
            filters["country"] = country
            break

    # Компания
    for alias, company in COMPANY_ALIASES.items():
        if alias in t:
            filters["company"] = company
            break

    # Тип работ
    for kw in sorted(WORKS_KEYWORDS.keys(), key=len, reverse=True):
        if kw in t:
            filters["work_tag"] = WORKS_KEYWORDS[kw]
            break

    # Бюджет (ищем числа с к/тыс/млн)
    import re
    budget_match = re.search(r'(\d+)\s*(?:к|тыс|тысяч|000)', t)
    if budget_match:
        filters["max_budget"] = int(budget_match.group(1)) * 1000
    budget_match_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:млн|миллион)', t)
    if budget_match_m:
        filters["max_budget"] = int(float(budget_match_m.group(1)) * 1_000_000)

    # Повторный/новый клиент
    repeat_keywords = ["повторный", "повторные", "постоянный", "постоянные",
                       "лояльный", "лояльные", "returning"]
    new_keywords = ["новый клиент", "новые клиенты", "первый раз", "новичок"]
    if any(kw in t for kw in repeat_keywords):
        filters["is_repeat_client"] = True
    elif any(kw in t for kw in new_keywords):
        filters["is_repeat_client"] = False

    return filters


def fetch_similar_projects(filters: dict, limit: int = 20) -> list[dict]:
    """Загружает проекты из kp_projects с фильтрацией."""
    params = {
        "select": "project_name,company,sphere,industry,client_price,project_type,works_tags,countries,country,country_raw,interview_depth,b2b_segment,is_repeat_client,research_goal",
        "limit": 1000,
        "order": "client_price.desc",
    }

    if filters.get("industry"):
        params["sphere"] = f"ilike.*{filters['industry']}*"

    if filters.get("company"):
        params["company"] = f"ilike.*{filters['company']}*"

    if filters.get("is_repeat_client") is not None:
        params["is_repeat_client"] = f"eq.{str(filters['is_repeat_client']).lower()}"

    r = requests.get(
        f"{SB_URL}/rest/v1/kp_projects",
        headers={k: v for k, v in SB_H.items() if k != "Content-Type"},
        params=params,
        timeout=20,
    )
    if r.status_code != 200:
        logger.error(f"Supabase error: {r.status_code} {r.text[:200]}")
        return []

    projects = r.json()

    # Post-filter: country
    if filters.get("country"):
        target_country = filters["country"].lower()
        projects = [
            p for p in projects
            if any(target_country in (c or "").lower() for c in (p.get("countries") or []))
            or target_country in (p.get("country") or "").lower()
            or target_country in (p.get("country_raw") or "").lower()
        ]

    # Post-filter: work_tag
    if filters.get("work_tag"):
        work_filter = filters["work_tag"].lower()
        projects = [
            p for p in projects
            if any(work_filter in (tag or "").lower() for tag in (p.get("works_tags") or []))
            or any(work_filter in (tag or "").lower() for tag in (p.get("tags") or []))
            or work_filter in (p.get("project_type") or "").lower()
        ]

    # Post-filter: segment
    if filters.get("segment"):
        seg = filters["segment"].lower()
        projects = [
            p for p in projects
            if seg in (p.get("b2b_segment") or "").lower()
            or seg in (p.get("interview_depth") or "").lower()
        ]

    # Post-filter: budget
    if filters.get("max_budget"):
        projects = [
            p for p in projects
            if p.get("client_price") and p["client_price"] <= filters["max_budget"]
        ]

    return projects[:limit]


def format_price(p) -> str:
    if not p:
        return "цена не указана"
    if p >= 1_000_000:
        return f"{p/1_000_000:.1f} млн ₽"
    elif p >= 1_000:
        return f"{p//1_000} тыс. ₽"
    return f"{p} ₽"


async def handle_similar_query(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Обрабатывает запрос на поиск похожих проектов."""
    await update.message.reply_text("🔍 Ищу похожие проекты...")
    try:
        filters = parse_similar_filters(text)
        projects = fetch_similar_projects(filters, limit=10)

        if not projects:
            filter_desc = []
            if filters.get("industry"):
                filter_desc.append(f"сфера: {filters['industry']}")
            if filters.get("country"):
                filter_desc.append(f"страна: {filters['country']}")
            if filters.get("work_tag"):
                filter_desc.append(f"тип работ: {filters['work_tag']}")
            if filters.get("max_budget"):
                filter_desc.append(f"бюджет до {format_price(filters['max_budget'])}")
            desc = ", ".join(filter_desc) if filter_desc else "заданным параметрам"
            await update.message.reply_text(
                f"Не нашёл проектов по {desc}. Попробуй изменить фильтры.",
                parse_mode="HTML"
            )
            return

        # Build filter description
        filter_parts = []
        if filters.get("industry"):
            filter_parts.append(f"сфера: <b>{filters['industry']}</b>")
        if filters.get("country"):
            filter_parts.append(f"страна: <b>{filters['country']}</b>")
        if filters.get("work_tag"):
            filter_parts.append(f"тип работ: <b>{filters['work_tag']}</b>")
        if filters.get("segment"):
            filter_parts.append(f"сегмент: <b>{filters['segment']}</b>")
        if filters.get("max_budget"):
            filter_parts.append(f"бюджет до <b>{format_price(filters['max_budget'])}</b>")
        if filters.get("company"):
            filter_parts.append(f"компания: <b>{filters['company']}</b>")
        if filters.get("is_repeat_client") is True:
            filter_parts.append("повторный клиент")
        elif filters.get("is_repeat_client") is False:
            filter_parts.append("новый клиент")

        filter_desc = " | ".join(filter_parts) if filter_parts else "все проекты"
        lines = [f"🔍 <b>Найдено проектов: {len(projects)}</b> ({filter_desc})\n"]

        for i, p in enumerate(projects[:10], 1):
            name = (p.get("project_name") or "—").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            company = (p.get("company") or "—").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            price = format_price(p.get("client_price"))
            sphere = p.get("sphere") or p.get("industry") or "—"
            works = ", ".join(p.get("works_tags") or []) or "—"
            goal = (p.get("research_goal") or "")[:100]

            lines.append(f"<b>{i}. {name}</b> ({company})")
            lines.append(f"   💰 {price} | 🏢 {sphere}")
            if works != "—":
                lines.append(f"   🔧 {works}")
            if goal:
                lines.append(f"   📋 {goal}...")
            lines.append("")

        response = "\n".join(lines)
        await update.message.reply_text(response, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Similar projects error: {e}")
        await update.message.reply_text("Ошибка при поиске проектов. Попробуй ещё раз.")

"""
Генерація інсайтів і порад для дизайнерів на основі
зв'язків між візуальними атрибутами та статистикою.
AI-резюме генерується через ENOT/Claude.
"""
import json
import requests
import pandas as pd
import numpy as np


# ─── Кореляції атрибутів з метриками ─────────────────────────────────────────

BINARY_ATTRS = ["mascot", "screenshot", "cta_button", "has_number", "ui_elements", "specific_benefit"]
CAT_ATTRS = ["bg_type", "offer_type", "style"]

MIN_GROUP_SIZE = 3   # мінімум креативів у групі для порівняння


def _mean_ci(series: pd.Series) -> tuple[float, float, int]:
    """(mean, std, n) для серії."""
    n = len(series)
    if n == 0:
        return 0.0, 0.0, 0
    return float(series.mean()), float(series.std()), n


def compute_attribute_stats(df: pd.DataFrame, visual: dict[str, dict]) -> dict:
    """
    Для кожного візуального атрибуту рахує середні CTR/CVR/CPA
    по групах (є / немає або категоріальні значення).
    Тільки included (>= min_impressions) креативи.
    """
    incl = df[~df["excluded"]].copy()

    # Приєднуємо візуальні атрибути
    for name, attrs in visual.items():
        for k, v in attrs.items():
            incl.loc[incl["name"] == name, f"vis_{k}"] = v if not isinstance(v, bool) else int(v)

    stats = {}

    # Бінарні атрибути
    for attr in BINARY_ATTRS:
        col = f"vis_{attr}"
        if col not in incl.columns:
            continue
        grp = incl.groupby(col)
        stats[attr] = {}
        for val, group in grp:
            if len(group) < MIN_GROUP_SIZE:
                continue
            stats[attr][bool(val)] = {
                "ctr":   _mean_ci(group["ctr_bayes"]),
                "cvr":   _mean_ci(group["cvr_bayes"]),
                "score": _mean_ci(group["composite_score"]),
                "n":     len(group),
            }

    # Категоріальні
    for attr in CAT_ATTRS:
        col = f"vis_{attr}"
        if col not in incl.columns:
            continue
        grp = incl.groupby(col)
        stats[attr] = {}
        for val, group in grp:
            if len(group) < MIN_GROUP_SIZE:
                continue
            stats[attr][val] = {
                "ctr":   _mean_ci(group["ctr_bayes"]),
                "cvr":   _mean_ci(group["cvr_bayes"]),
                "score": _mean_ci(group["composite_score"]),
                "n":     len(group),
            }

    return stats


def generate_insights(df: pd.DataFrame, visual: dict[str, dict]) -> dict:
    """
    Повертає словник з інсайтами:
      - top10 / bottom10 за composite_score
      - attribute_stats
      - ctr_drivers: атрибути з найвищим CTR
      - cvr_drivers: атрибути з найвищим CVR
      - traps: CTR high, CVR low
      - ab_pairs: пари схожих назв
    """
    incl = df[~df["excluded"]].copy()
    top10    = df[~df["excluded"]].nlargest(10, "composite_score")["name"].tolist()
    bottom10 = df[~df["excluded"]].nsmallest(10, "composite_score")["name"].tolist()

    attr_stats = compute_attribute_stats(df, visual)

    # Драйвери CTR: знаходимо атрибут/значення з найвищим середнім CTR
    ctr_drivers = []
    for attr, groups in attr_stats.items():
        for val, s in groups.items():
            ctr_drivers.append({"attr": attr, "val": val, "ctr_mean": s["ctr"][0], "n": s["n"]})
    ctr_drivers.sort(key=lambda x: -x["ctr_mean"])

    # Драйвери CVR
    cvr_drivers = []
    for attr, groups in attr_stats.items():
        for val, s in groups.items():
            cvr_drivers.append({"attr": attr, "val": val, "cvr_mean": s["cvr"][0], "n": s["n"]})
    cvr_drivers.sort(key=lambda x: -x["cvr_mean"])

    # Пастки: top-25% CTR але bottom-25% CVR
    if len(incl) >= 8:
        ctr_q75 = incl["ctr_bayes"].quantile(0.75)
        cvr_q25 = incl["cvr_bayes"].quantile(0.25)
        traps = incl[
            (incl["ctr_bayes"] >= ctr_q75) & (incl["cvr_bayes"] <= cvr_q25)
        ]["name"].tolist()
    else:
        traps = []

    # A/B пари: назви що відрізняються тільки в кінці (_1/_2, _black/_white, тощо)
    names = list(df["name"])
    ab_pairs = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            base1 = n1.rsplit("_", 1)[0] if "_" in n1 else n1
            base2 = n2.rsplit("_", 1)[0] if "_" in n2 else n2
            if base1 == base2 and base1:
                ab_pairs.append((n1, n2))

    return {
        "top10":        top10,
        "bottom10":     bottom10,
        "attr_stats":   attr_stats,
        "ctr_drivers":  ctr_drivers[:5],
        "cvr_drivers":  cvr_drivers[:5],
        "traps":        traps,
        "ab_pairs":     ab_pairs[:10],
        "ai_summary":   "",   # заповнюється окремо через generate_ai_summary()
    }


#AI-резюме через ENOT

def generate_ai_summary(
    insights: dict,
    df: pd.DataFrame,
    visual: dict[str, dict],
    enot_key: str,
    enot_url: str,
) -> str:
    """
    Передає статистику атрибутів + топ/анти-топ в Claude і отримує
    текстове резюме з порадами для дизайнерів українською мовою.
    Результат зберігається в insights["ai_summary"].
    """
    if not enot_key:
        return ""

    incl = df[~df["excluded"]].copy()

    # Формуємо компактний контекст для Claude
    # 1. Топ-5 і анти-топ-5 з атрибутами
    def creative_summary(name: str) -> dict:
        row = incl[incl["name"] == name]
        vis = visual.get(name, {})
        if not len(row):
            return {"name": name}
        r = row.iloc[0]
        return {
            "name":    name,
            "score":   round(float(r["composite_score"]), 2),
            "ctr":     f"{r['ctr_bayes']:.2%}",
            "cvr":     f"{r['cvr_bayes']:.2%}",
            "cpa":     f"${r['cpa']:.2f}" if r.get("cpa") else "—",
            "leads":   int(r["leads"]),
            "label":   r["label"],
            **{k: v for k, v in vis.items() if k != "text_lines"},
        }

    top5    = [creative_summary(n) for n in insights["top10"][:5]]
    bottom5 = [creative_summary(n) for n in insights["bottom10"][:5]]

    # 2. Статистика атрибутів (спрощена)
    attr_summary = []
    for attr, groups in insights["attr_stats"].items():
        for val, s in groups.items():
            attr_summary.append({
                "attr":  attr,
                "value":  str(val),
                "avg_ctr":   f"{s['ctr'][0]:.2%}",
                "avg_cvr":   f"{s['cvr'][0]:.2%}",
                "avg_score": round(s["score"][0], 2),
                "n":         s["n"],
            })
    attr_summary.sort(key=lambda x: -float(x["avg_score"]))

    # 3. Пастки
    traps = insights.get("traps", [])

    context = json.dumps({
        "total_creatives":  len(incl),
        "top5_creatives":   top5,
        "bottom5_creatives": bottom5,
        "attribute_stats":  attr_summary[:20],
        "traps_high_ctr_low_cvr": traps,
        "ctr_drivers":  insights["ctr_drivers"],
        "cvr_drivers":  insights["cvr_drivers"],
    }, ensure_ascii=False, indent=2)

    schema = json.dumps({
        "type": "object",
        "properties": {
            "management_summary": {"type": "string"},
            "top_creative_why":   {"type": "string"},
            "traps_analysis":     {"type": "string"},
            "recommendations":    {"type": "string"},
        },
        "required": ["management_summary", "top_creative_why", "traps_analysis", "recommendations"],
    })

    system_prompt = (
        "Ти — senior performance-маркетолог і аналітик креативів для monobank Business. "
        "Пишеш висновки для двох аудиторій: керівництва (бізнес-результати, цифри, ефективність бюджету) "
        "та команди дизайнерів (конкретні рекомендації що робити з креативами). "
        "Мова: українська. Тон: діловий, конкретний, без води. "
        "Спирайся ТІЛЬКИ на надані цифри — не вигадуй."
    )

    user_prompt = f"""Ось дані аналізу {len(incl)} рекламних креативів monobank Business (YO/FOP/Acquiring).

Дані:
{context}

Напиши структурований аналіз у 4 розділах:

1. management_summary — Висновки для керівництва. Що загалом відбувається з ефективністю креативів? Який сегмент (YO/FOP/Acquiring) показує найкращі результати? Де найвищий Score і найнижчий CPA? Конкретні цифри. 3-5 речень.

2. top_creative_why — Чому топ-5 креативів працюють? Що між ними спільного у візуальних атрибутах, стилі, оферах? Що їх відрізняє від анти-топу?

3. traps_analysis — Пастки та проблеми: де є конфлікт між CTR і CVR (привертають увагу, але не конвертують)? Назви конкретні креативи з цифрами.

4. recommendations — 4-5 конкретних порад дизайнерам. Що робити більше, що прибрати, що спробувати. Формат: короткий заголовок + 1-2 речення пояснення."""

    try:
        resp = requests.post(
            enot_url,
            headers={"X-API-Key": enot_key},
            data={
                "prompt":     system_prompt,
                "user_input": user_prompt,
                "schema":     schema,
                "model":      "claude-3.7",
            },
            timeout=60,
        )
        if resp.status_code == 200:
            result = resp.json().get("result", {})
            insights["ai_summary"] = result
            return result
        else:
            insights["ai_summary"] = {"error": f"API {resp.status_code}"}
            return {}
    except Exception as e:
        insights["ai_summary"] = {"error": str(e)}
        return {}

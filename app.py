"""
app.py — Streamlit UI для аналізу рекламних креативів Meta Ads.

Запуск:
    streamlit run app.py

При першому запуску або натисканні "Оновити дані" — тягне з Meta API,
аналізує зображення через ENOT, зберігає кеш. Наступні відкриття — з кешу.
"""
import sys
import pickle
import base64
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import META_ACCESS_TOKEN, META_API_BASE, META_AD_ACCOUNT, ENOT_API_KEY, ENOT_API_URL, SCORING
from data.meta_api import load_or_fetch
from logic.visual_analyzer import analyze_all, load_cache as load_visual_cache
from logic.scoring import build_scores
from logic.insights import generate_insights, generate_ai_summary

CACHE_RAW    = ROOT / "data" / "cache" / "ads_raw.pkl"
CACHE_VISUAL = ROOT / "data" / "cache" / "visual_attrs.pkl"

# ─── Константи для відображення ──────────────────────────────────────────────

OFFER_LABELS = {
    "rate": "ставка", "fx": "валюта", "payroll": "ЗП проект",
    "cashback": "кешбек", "review": "відгук", "payments": "платежі",
    "kep": "КЕП", "acquiring": "еквайринг", "general": "загальне", "other": "інше",
}
STYLE_LABELS = {
    "minimal": "мінімал", "illustrated": "ілюстрація",
    "product": "продукт", "card": "картка", "mixed": "mixed", "video": "відео",
}
BG_LABELS = {
    "black": "чорний", "dark_blue": "темно-синій", "navy": "темно-синій (navy)",
    "dark_gradient": "темний градієнт", "chalk": "крейда",
    "white": "білий", "light_grey": "світло-сірий", "light_blue": "світло-блакитний",
    "other": "інший",
}
LABEL_COLORS = {"YO": "🟣", "FOP": "🔵", "Acquiring": "🟢"}
ATTR_ICONS = {
    "mascot": "🐱", "screenshot": "📱", "cta_button": "🔘",
    "has_number": "🔢", "ui_elements": "🧩", "specific_benefit": "✅",
}


# ─── Міграція з /tmp/ ────────────────────────────────────────────────────────

def _migrate_tmp():
    meta_full   = Path("/tmp/meta_full.pkl")
    cache_dir   = ROOT / "data" / "cache"
    ads_cache   = cache_dir / "ads_raw.pkl"
    vis_cache   = cache_dir / "visual_attrs.pkl"
    if not meta_full.exists():
        return
    if not ads_cache.exists():
        with open(meta_full, "rb") as f:
            data = pickle.load(f)
        agg = data.get("agg", {})
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(ads_cache, "wb") as f:
            pickle.dump(agg, f)
    if not vis_cache.exists():
        with open(meta_full, "rb") as f:
            data = pickle.load(f)
        visual = data.get("visual", {})
        if visual:
            with open(vis_cache, "wb") as f:
                pickle.dump(visual, f)


# ─── Завантаження даних (з кешу або API) ─────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(force_api: bool = False, force_vis: bool = False):
    _migrate_tmp()
    agg    = load_or_fetch(META_ACCESS_TOKEN, META_API_BASE, META_AD_ACCOUNT, force=force_api)
    visual = analyze_all(agg, ENOT_API_KEY, ENOT_API_URL, workers=3, force=force_vis)
    df     = build_scores(
        agg,
        min_impressions=SCORING["min_impressions"],
        ctr_prior_n=SCORING["ctr_prior_n"],
        cvr_prior_n=SCORING["cvr_prior_n"],
        w_ctr=SCORING["weight_ctr"],
        w_cvr=SCORING["weight_cvr"],
        w_cpa=SCORING["weight_cpa"],
    )
    insights = generate_insights(df, visual)
    return agg, visual, df, insights


# ─── Хелпери UI ──────────────────────────────────────────────────────────────

def score_badge(score: float) -> str:
    color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
    return f":{color}[**{score:.2f}**]"


def attr_tags(vis: dict) -> str:
    tags = []
    for attr, icon in ATTR_ICONS.items():
        if vis.get(attr):
            tags.append(icon)
    ot = OFFER_LABELS.get(vis.get("offer_type", ""), "")
    st_ = STYLE_LABELS.get(vis.get("style", ""), "")
    if ot:
        tags.append(f"`{ot}`")
    if st_:
        tags.append(f"`{st_}`")
    return " ".join(tags)


def thumbnail_html(b64: str, size: int = 72) -> str:
    if not b64:
        return f'<div style="width:{size}px;height:{size}px;background:#1e293b;border-radius:6px"></div>'
    return f'<img src="{b64}" style="width:{size}px;height:{size}px;object-fit:cover;border-radius:6px">'


# ─── Сторінки ────────────────────────────────────────────────────────────────

def page_table(df: pd.DataFrame, visual: dict, agg: dict):
    st.subheader("Таблиця креативів")

    # Фільтри
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        label_filter = st.selectbox("Сегмент", ["Всі", "YO", "FOP", "Acquiring"])
    with col2:
        show_excl = st.checkbox("Показати виключені (<500 показів)", value=False)
    with col3:
        search = st.text_input("Пошук по назві", placeholder="Введіть назву...")

    sort_col = st.selectbox(
        "Сортувати за",
        ["composite_score", "ctr_bayes", "cvr_bayes", "cpa", "impressions", "results", "spend"],
        format_func=lambda x: {
            "composite_score": "Composite Score",
            "ctr_bayes":       "CTR (Bayesian)",
            "cvr_bayes":       "Post-click CVR (Bayesian)",
            "cpa":             "CPA",
            "impressions":     "Покази",
            "results":         "Results",
            "spend":           "Spend",
        }.get(x, x)
    )

    # Фільтрація
    view = df.copy()
    if label_filter != "Всі":
        view = view[view["label"] == label_filter]
    if not show_excl:
        view = view[~view["excluded"]]
    if search:
        view = view[view["name"].str.contains(search, case=False, na=False)]
    asc = sort_col == "cpa"
    view = view.sort_values(sort_col, ascending=asc, na_position="last")

    st.caption(f"Показано: {len(view)} креативів")

    # Заголовки таблиці
    hdr = st.columns([1, 3, 2, 1, 1, 1, 1, 1, 1])
    for col, label in zip(hdr, ["Фото", "Назва", "Атрибути", "Score", "CTR", "CVR", "Results", "CPA", "Spend"]):
        col.markdown(f"**{label}**")
    st.divider()

    for _, row in view.iterrows():
        vis  = visual.get(row["name"], {})
        excl = row.get("excluded", False)

        c_img, c_info, c_attrs, c_score, c_ctr, c_cvr, c_res, c_cpa, c_spend = st.columns([1, 3, 2, 1, 1, 1, 1, 1, 1])

        with c_img:
            b64 = row.get("thumb_b64") or agg.get(row["name"], {}).get("thumb_b64", "")
            st.markdown(thumbnail_html(b64, 72), unsafe_allow_html=True)

        with c_info:
            lbl = row.get("label", "?")
            obj = row.get("objective", "")
            st.markdown(f"**{row['name']}**")
            obj_line = f"{LABEL_COLORS.get(lbl,'⚪')} `{lbl}`"
            if obj:
                obj_line += f" · `{obj}`"
            st.markdown(obj_line)

        with c_attrs:
            if vis:
                lines = []
                # Стиль, фон, офер
                style  = STYLE_LABELS.get(vis.get("style", ""), "")
                bg     = BG_LABELS.get(vis.get("bg_type", ""), "")
                offer  = OFFER_LABELS.get(vis.get("offer_type", ""), "")
                if style:
                    lines.append(f"Стиль: **{style}**")
                if bg:
                    lines.append(f"Фон: **{bg}**")
                if offer:
                    lines.append(f"Офер: **{offer}**")
                # Бінарні атрибути — тільки ті що True
                BINARY_LABELS = {
                    "mascot":           "Маскот",
                    "screenshot":       "Скріншот",
                    "cta_button":       "CTA кнопка",
                    "has_number":       "Цифри",
                    "ui_elements":      "UI елементи",
                    "specific_benefit": "Конкретна вигода",
                }
                present = [label for key, label in BINARY_LABELS.items() if vis.get(key)]
                if present:
                    lines.append(", ".join(present))
                st.markdown("  \n".join(lines) if lines else "—")
            else:
                st.caption("—")

        with c_score:
            if excl:
                st.caption("—")
            else:
                st.markdown(score_badge(row.get("composite_score", 0.0)))

        with c_ctr:
            ctr = row.get("ctr_bayes", 0)
            st.metric(label="", value=f"{ctr:.2%}")

        with c_cvr:
            cvr = row.get("cvr_bayes", 0)
            st.metric(label="", value=f"{cvr:.2%}")

        with c_res:
            results = row.get("results", 0)
            rat     = row.get("results_action_type", "")
            rat_short = rat.split(".")[-1] if rat else "—"
            st.metric(label="", value=f"{int(results)}")
            if rat_short and rat_short != "—":
                st.caption(rat_short)

        with c_cpa:
            cpa = row.get("cpa")
            st.metric(label="", value=f"${cpa:.2f}" if cpa else "—")

        with c_spend:
            spend = row.get("spend", 0)
            st.metric(label="", value=f"${spend:.0f}")

        st.divider()


def page_insights(df: pd.DataFrame, visual: dict, insights: dict):
    st.subheader("Інсайти")

    # ── Пояснення скорингової логіки ──────────────────────────────────────────
    with st.expander("Як рахується Composite Score — для керівництва та дизайнерів", expanded=False):
        st.markdown("""
**Composite Score** — єдина оцінка від 0 до 1, яка показує наскільки ефективний креатив у воронці.

Складається з трьох компонент:

| Метрика | Вага | Що вимірює |
|---|---|---|
| **CTR** (кліки / покази) | 35% | Чи привертає оголошення увагу — чи хочуть люди клікнути |
| **CVR** (конверсії / кліки) | 35% | Якість трафіку — чи стають кліки результатами (лідами, заявками) |
| **CPA** (витрати / конверсія) | 30% | Ефективність бюджету — скільки коштує один результат |

Кожна метрика переводиться у **перцентильний ранг** (0–1) серед усіх креативів з ≥500 показів,
після чого зважено сумується. Score **0.7+** — зелений (топ), **0.4–0.7** — жовтий, **< 0.4** — червоний.

> CTR і CVR згладжені байєсівським методом — нові креативи з малою вибіркою не завищуються штучно.
        """)

    st.divider()

    # ── Топ / Анти-топ ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Топ-10 за Composite Score")
        for i, name in enumerate(insights.get("top10", []), 1):
            row = df[df["name"] == name]
            if not len(row):
                continue
            r = row.iloc[0]
            st.markdown(
                f"{i}. **{name}**  \n"
                f"   Score: `{r['composite_score']:.2f}` · CTR: `{r['ctr_bayes']:.2%}` · "
                f"CVR: `{r['cvr_bayes']:.2%}` · CPA: `{'$'+str(round(r['cpa'],2)) if r.get('cpa') else '—'}`"
            )

    with col2:
        st.markdown("#### Анти-топ-10")
        for i, name in enumerate(insights.get("bottom10", []), 1):
            row = df[df["name"] == name]
            if not len(row):
                continue
            r = row.iloc[0]
            st.markdown(
                f"{i}. **{name}**  \n"
                f"   Score: `{r['composite_score']:.2f}` · CTR: `{r['ctr_bayes']:.2%}` · "
                f"CVR: `{r['cvr_bayes']:.2%}` · CPA: `{'$'+str(round(r['cpa'],2)) if r.get('cpa') else '—'}`"
            )

    # ── AI-аналіз ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### AI-аналіз")
    st.caption("Claude аналізує дані і надає висновки для керівництва та рекомендації для дизайнерів")

    ai = st.session_state.get("ai_summary")

    if not ai:
        if st.button("Згенерувати AI-аналіз", type="primary"):
            with st.spinner("Claude аналізує дані (~30с)..."):
                result = generate_ai_summary(insights, df, visual, ENOT_API_KEY, ENOT_API_URL)
                st.session_state["ai_summary"] = result
                ai = result

    if ai and not (isinstance(ai, dict) and "error" in ai):
        SECTION_LABELS = {
            "management_summary": "Висновки для керівництва",
            "top_creative_why":   "Чому топ-креативи працюють",
            "traps_analysis":     "Пастки: кліки є, конверсій немає",
            "recommendations":    "Рекомендації дизайнерам",
        }
        for key, label in SECTION_LABELS.items():
            text = ai.get(key, "") if isinstance(ai, dict) else ""
            if text:
                with st.expander(label, expanded=(key in ("management_summary", "recommendations"))):
                    st.markdown(text)
        if st.button("Оновити AI-аналіз"):
            del st.session_state["ai_summary"]
            st.rerun()
    elif isinstance(ai, dict) and "error" in ai:
        st.error(f"Помилка генерації: {ai['error']}")
        if st.button("Спробувати знову"):
            del st.session_state["ai_summary"]
            st.rerun()
            if text:
                with st.expander(label, expanded=(key == "recommendations")):
                    st.markdown(text)
        if st.button("Оновити AI-аналіз"):
            del st.session_state["ai_summary"]
            st.rerun()
    elif isinstance(ai, dict) and "error" in ai:
        st.error(f"Помилка генерації: {ai['error']}")
        if st.button("Спробувати знову"):
            del st.session_state["ai_summary"]
            st.rerun()


def page_overview(df: pd.DataFrame, visual: dict):
    st.subheader("Огляд")

    incl = df[~df["excluded"]]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Унікальних креативів", len(df))
    c2.metric("В рейтингу", len(incl))
    c3.metric("Виключено", int(df["excluded"].sum()))
    c4.metric("Проаналізовано", len(visual))
    c5.metric("Загальний spend", f"${df['spend'].sum():,.0f}")
    c6.metric("Всього Results", f"{int(df['results'].sum()):,}")

    st.divider()

    # Розподіл по сегментах
    st.markdown("#### Розподіл по сегментах")
    seg_stats = incl.groupby("label").agg(
        креативів=("name", "count"),
        spend=("spend", "sum"),
        leads=("leads", "sum"),
        avg_score=("composite_score", "mean"),
    ).round(2)
    st.dataframe(seg_stats, use_container_width=True)

    st.divider()

    # Топ-5 по score з фільтром сегменту
    t_col1, t_col2 = st.columns([2, 5])
    with t_col1:
        seg_filter = st.selectbox(
            "Сегмент",
            ["Всі", "YO", "FOP", "Acquiring"],
            key="overview_seg_filter",
        )
    st.markdown("#### Топ-5 за Composite Score")
    filtered = incl if seg_filter == "Всі" else incl[incl["label"] == seg_filter]
    top5 = filtered.nlargest(5, "composite_score")[
        ["name", "label", "objective", "composite_score", "ctr_bayes", "cvr_bayes",
         "results", "results_action_type", "cpa", "spend"]
    ].rename(columns={
        "composite_score":     "Score",
        "ctr_bayes":           "CTR",
        "cvr_bayes":           "CVR",
        "results":             "Results",
        "results_action_type": "Results type",
    })
    if len(top5) == 0:
        st.caption("Немає креативів для обраного сегменту.")
    else:
        st.dataframe(top5, use_container_width=True, hide_index=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Meta Ads Creative Analysis",
        layout="wide"
    )

    st.title("Meta Ads Creative Analysis")
    st.caption("monobank Business · YO / FOP / Acquiring · Дані з Meta API")

    # Сайдбар
    with st.sidebar:
        st.markdown("### Керування даними")

        cache_exists = CACHE_RAW.exists()
        if cache_exists:
            import os, datetime
            mtime = os.path.getmtime(CACHE_RAW)
            updated = datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
            st.success(f"Кеш є · оновлено {updated}")
        else:
            st.warning("Кеш відсутній — буде завантажено з Meta API")

        force_api = st.button("🔄 Оновити дані з Meta API", use_container_width=True)
        force_vis = st.button("🔍 Переаналізувати зображення", use_container_width=True)

        if force_api or force_vis:
            st.cache_data.clear()

        st.divider()
        st.markdown("### Навігація")
        page = st.radio("Навігація", ["Огляд", "Таблиця креативів", "Інсайти"], label_visibility="collapsed")

    # Завантаження
    with st.spinner("Завантажую дані..."):
        try:
            agg, visual, df, insights = load_data(
                force_api=force_api,
                force_vis=force_vis,
            )
        except Exception as e:
            st.error(f"Помилка завантаження: {e}")
            st.stop()

    # Сторінки
    if page == "Огляд":
        page_overview(df, visual)
    elif page == "Таблиця креативів":
        page_table(df, visual, agg)
    elif page == "Інсайти":
        page_insights(df, visual, insights)


if __name__ == "__main__":
    main()

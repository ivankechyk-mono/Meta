"""
Використання:
  python run_analysis.py              # з кешу якщо є
  python run_analysis.py --force-api  # оновити дані з Meta API
  python run_analysis.py --force-vis  # переаналізувати всі зображення
"""
import sys
import pickle
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import META_ACCESS_TOKEN, META_API_BASE, META_AD_ACCOUNT, ENOT_API_KEY, ENOT_API_URL, SCORING
from data.meta_api import load_or_fetch
from logic.visual_analyzer import analyze_all
from logic.scoring import build_scores
from logic.insights import generate_insights, generate_ai_summary

CACHE_DIR = ROOT / "data" / "cache"


def migrate_tmp_cache():
    meta_full  = Path("/tmp/meta_full.pkl")
    ads_cache  = CACHE_DIR / "ads_raw.pkl"
    vis_cache  = CACHE_DIR / "visual_attrs.pkl"
    if not meta_full.exists():
        return
    if not ads_cache.exists():
        print("Мігруємо ads_raw з /tmp/meta_full.pkl...")
        with open(meta_full, "rb") as f:
            data = pickle.load(f)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(ads_cache, "wb") as f:
            pickle.dump(data.get("agg", {}), f)
        print(f"  → {ads_cache}")
    if not vis_cache.exists():
        with open(meta_full, "rb") as f:
            data = pickle.load(f)
        visual = data.get("visual", {})
        if visual:
            with open(vis_cache, "wb") as f:
                pickle.dump(visual, f)
            print(f"  → {vis_cache} ({len(visual)} записів)")


def main():
    parser = argparse.ArgumentParser(description="Meta Ads data collection & visual analysis")
    parser.add_argument("--force-api", action="store_true", help="Оновити дані з Meta API")
    parser.add_argument("--force-vis", action="store_true", help="Переаналізувати всі зображення")
    args = parser.parse_args()

    migrate_tmp_cache()

    print("\n КРОК 1: Дані Meta API ")
    agg = load_or_fetch(META_ACCESS_TOKEN, META_API_BASE, META_AD_ACCOUNT, force=args.force_api)
    print(f"Креативів: {len(agg)}")

    print("\n КРОК 2: Візуальний аналіз")
    visual = analyze_all(agg, ENOT_API_KEY, ENOT_API_URL, workers=3, force=args.force_vis)
    print(f"Проаналізовано: {len(visual)}/{len(agg)}")

    print("\n КРОК 3: Scoring (preview)")
    df = build_scores(agg, min_impressions=SCORING["min_impressions"])
    incl = df[~df["excluded"]]
    print(f"В рейтингу: {len(incl)}, виключено: {df['excluded'].sum()}")
    print("\nТоп-5:")
    print(incl.nlargest(5, "composite_score")[["name", "composite_score", "ctr_bayes", "cvr_bayes"]].to_string(index=False))

    print("\n КРОК 4: AI-резюме")
    insights = generate_insights(df, visual)
    ai = generate_ai_summary(insights, df, visual, ENOT_API_KEY, ENOT_API_URL)
    if ai and "error" not in ai:
        print("AI-резюме згенеровано")
        for key, text in ai.items():
            print(f"\n[{key}]\n{text[:200]}...")
    else:
        print(f"AI-резюме недоступне: {ai}")

    print("\nГотово. Запусти: streamlit run app.py")


if __name__ == "__main__":
    main()

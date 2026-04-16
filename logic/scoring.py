"""
Скорингова модель для рекламних креативів.

Компоненти воронки:
  CTR          = clicks / impressions          (привернення уваги)
  Post-click CVR = leads / clicks             (якість трафіку)
  CPA          = spend / leads                (ефективність витрат)

Байєсове згладжування (empirical Bayes) стабілізує оцінки при малій вибірці.
Wilson interval дає 95% confidence interval для CTR та CVR.
"""
import math
import numpy as np
import pandas as pd
from scipy import stats


# ─── Wilson confidence interval ──────────────────────────────────────────────

def wilson_ci(successes: float, trials: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval для пропорції. Повертає (lower, upper)."""
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    spread = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# ─── Bayesian smoothing ───────────────────────────────────────────────────────

def bayesian_rate(successes: float, trials: float, prior_rate: float, prior_n: float) -> float:
    """
    Empirical Bayes: (alpha + successes) / (alpha + beta + trials)
    де alpha = prior_rate * prior_n, beta = (1 - prior_rate) * prior_n
    """
    alpha = prior_rate * prior_n
    beta = (1 - prior_rate) * prior_n
    return (alpha + successes) / (alpha + beta + trials)


# Нормалізація в [0,1]

def _percentile_score(series: pd.Series) -> pd.Series:
    """Ранжує значення у відсоткові (0–1), вищий = кращий."""
    return series.rank(pct=True, method="average")


def _percentile_score_inv(series: pd.Series) -> pd.Series:
    """Інвертований percentile rank (нижча CPA = кращий скор)."""
    return 1 - series.rank(pct=True, method="average")


# ─── Основна функція ─────────────────────────────────────────────────────────

def build_scores(
    agg: dict[str, dict],
    min_impressions: int = 500,
    ctr_prior_n:    float = 1000.0,
    cvr_prior_n:    float = 100.0,
    w_ctr: float = 0.35,
    w_cvr: float = 0.35,
    w_cpa: float = 0.30,
) -> pd.DataFrame:
    """
    Будує DataFrame з усіма метриками та composite score.

    Повертає DataFrame відсортований за composite_score DESC.
    Колонка `excluded` = True для креативів з < min_impressions.
    """
    rows = []
    for name, r in agg.items():
        rows.append({
            "name":                name,
            "label":               r.get("label", "?"),
            "objective":           r.get("objective", ""),
            "results_action_type": r.get("results_action_type", ""),
            "results":             r.get("results", 0.0),
            "impressions":         r.get("impressions", 0),
            "clicks":              r.get("clicks", 0),
            "leads":               r.get("leads", 0.0),
            "custom_conv":         r.get("custom_conv", 0.0),
            "spend":               r.get("spend", 0.0),
            "thumbnail_url":       r.get("thumbnail_url", ""),
            "thumb_b64":           r.get("thumb_b64", ""),
        })

    df = pd.DataFrame(rows)

    # Основна метрика конверсій: results (правильний тип по objective), fallback leads/custom_conv
    df["conversions"] = df.apply(
        lambda x: x["results"] if x["results"] > 0
        else (x["leads"] if x["leads"] > 0 else x["custom_conv"]),
        axis=1,
    )

    # Прапор виключення
    df["excluded"] = df["impressions"] < min_impressions

    # ── Empirical Bayes priors (з повного датасету) ──
    main = df[~df["excluded"]]
    prior_ctr = main["clicks"].sum() / main["impressions"].sum() if main["impressions"].sum() > 0 else 0.02
    valid_cvr  = main[main["clicks"] > 0]
    prior_cvr  = valid_cvr["conversions"].sum() / valid_cvr["clicks"].sum() if len(valid_cvr) > 0 else 0.005

    # ── Байєсові оцінки ──
    df["ctr_bayes"] = df.apply(
        lambda x: bayesian_rate(x["clicks"], x["impressions"], prior_ctr, ctr_prior_n), axis=1
    )
    df["cvr_bayes"] = df.apply(
        lambda x: bayesian_rate(x["conversions"], x["clicks"], prior_cvr, cvr_prior_n)
        if x["clicks"] > 0 else prior_cvr, axis=1
    )
    df["cpa"] = df.apply(
        lambda x: x["spend"] / x["conversions"] if x["conversions"] > 0 else float("inf"), axis=1
    )

    # ── Confidence intervals (Wilson) ──
    df[["ctr_ci_lo", "ctr_ci_hi"]] = df.apply(
        lambda x: pd.Series(wilson_ci(x["clicks"], x["impressions"])), axis=1
    )
    df[["cvr_ci_lo", "cvr_ci_hi"]] = df.apply(
        lambda x: pd.Series(wilson_ci(x["conversions"], x["clicks"])) if x["clicks"] > 0
        else pd.Series((0.0, 0.0)), axis=1
    )

    # ── Composite score (тільки для included) ──
    incl = df[~df["excluded"]].copy()
    if len(incl) > 0:
        incl["score_ctr"] = _percentile_score(incl["ctr_bayes"])
        incl["score_cvr"] = _percentile_score(incl["cvr_bayes"])
        # CPA: inf → найгірший
        cpa_finite = incl["cpa"].replace(float("inf"), incl["cpa"].replace(float("inf"), np.nan).max() * 2)
        incl["score_cpa"] = _percentile_score_inv(cpa_finite)

        incl["composite_score"] = (
            w_ctr * incl["score_ctr"] +
            w_cvr * incl["score_cvr"] +
            w_cpa * incl["score_cpa"]
        ).round(4)

        df = df.merge(
            incl[["name", "score_ctr", "score_cvr", "score_cpa", "composite_score"]],
            on="name", how="left"
        )
    else:
        df["score_ctr"] = df["score_cvr"] = df["score_cpa"] = df["composite_score"] = 0.0

    df["composite_score"] = df["composite_score"].fillna(0.0)

    # Округлення для відображення
    for col in ["ctr_bayes", "cvr_bayes", "ctr_ci_lo", "ctr_ci_hi", "cvr_ci_lo", "cvr_ci_hi"]:
        df[col] = df[col].round(4)
    df["cpa"] = df["cpa"].apply(lambda x: round(x, 2) if x != float("inf") else None)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

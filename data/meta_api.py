"""
Meta Graph API — завантаження кампаній, ads, статистики та thumbnail.
"""
import re
import time
import pickle
import base64
import requests
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_RAW = CACHE_DIR / "ads_raw.pkl"


# ─── Мітки кампаній ──────────────────────────────────────────────────────────

MARK_PATTERNS = {
    label: re.compile(re.escape(mark), re.IGNORECASE)
    for label, mark in {
        "YO": "_pr_mpc_reg_YO",
        "FOP": "_pr_mpc_reg_FOP",
        "Acquiring": "_pr_mpc_reg_Acquiring",
    }.items()
}


# ─── API helpers ─────────────────────────────────────────────────────────────

def _get(url, params, timeout=30, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 524) and attempt < retries - 1:
                wait = 2 ** attempt * (10 if status == 429 else 1)  # 429 → довший backoff
                print(f"  Meta API {status}, retry {attempt+1}/{retries-1} через {wait}s...")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Meta API timeout, retry {attempt+1}/{retries-1} через {wait}s...")
                time.sleep(wait)
                continue
            raise


def get_target_campaigns(token: str, base_url: str, acc_id: str) -> dict[str, dict]:
    """Повертає {campaign_id: {label, name, objective}} для цільових кампаній."""
    result = {}
    url = f"{base_url}/{acc_id}/campaigns"
    params = {
        "access_token": token,
        "fields": "id,name,status,objective",
        "limit": 200,
    }
    while url:
        data = _get(url, params)
        for c in data.get("data", []):
            for label, pat in MARK_PATTERNS.items():
                if pat.search(c["name"]):
                    result[c["id"]] = {
                        "label":     label,
                        "name":      c["name"],
                        "objective": c.get("objective", ""),
                    }
                    break
        url = data.get("paging", {}).get("next")
        params = {}
    return result


# Маппінг objective → action_type який Meta показує як "Results"
OBJECTIVE_RESULTS_ACTION = {
    "LEAD_GENERATION":    ["lead", "onsite_conversion.lead_grouped"],
    "OUTCOME_LEADS":      ["lead", "offsite_conversion.fb_pixel_lead", "onsite_conversion.lead_grouped"],
    "CONVERSIONS":        ["offsite_conversion.fb_pixel_purchase", "offsite_conversion.fb_pixel_lead",
                           "offsite_conversion.fb_pixel_custom"],
    "OUTCOME_SALES":      ["offsite_conversion.fb_pixel_purchase", "offsite_conversion.fb_pixel_custom"],
    "LINK_CLICKS":        ["link_click"],
    "REACH":              ["reach"],
    "BRAND_AWARENESS":    ["reach"],
    "VIDEO_VIEWS":        ["video_view"],
    "MESSAGES":           ["onsite_conversion.messaging_first_reply"],
    "POST_ENGAGEMENT":    ["post_engagement"],
    "APP_INSTALLS":       ["mobile_app_install"],
    "OUTCOME_TRAFFIC":    ["link_click", "landing_page_view"],
    "OUTCOME_ENGAGEMENT": ["post_engagement", "video_view"],
    "OUTCOME_APP_PROMOTION": ["mobile_app_install"],
    "OUTCOME_AWARENESS":  ["reach"],
}


def get_all_ads(token: str, base_url: str, acc_id: str, campaign_ids: set) -> list[dict]:
    """Завантажує всі ads акаунту з пагінацією, фільтрує по campaign_ids."""
    ads = []
    url = f"{base_url}/{acc_id}/ads"
    params = {
        "access_token": token,
        "fields": (
            "id,name,status,campaign_id,"
            "creative{id,name,thumbnail_url,image_url,asset_feed_spec},"
            "insights.date_preset(maximum){"
            "spend,impressions,clicks,reach,"
            "actions,cost_per_action_type"
            "}"
        ),
        "limit": 100,
    }
    page = 0
    while url:
        page += 1
        try:
            data = _get(url, params, timeout=60)
        except Exception as e:
            print(f"  Сторінка {page}: критична помилка {e} — зупиняємо пагінацію")
            break
        batch = data.get("data", [])
        added = 0
        for ad in batch:
            if ad.get("campaign_id") in campaign_ids:
                ads.append(ad)
                added += 1
        print(f"  Сторінка {page}: {len(batch)} ads, відібрано {added}")
        next_url = data.get("paging", {}).get("next")
        url = next_url
        params = {}
        time.sleep(0.3)
    return ads


def get_full_image_urls(token: str, base_url: str, acc_id: str, hashes: list[str]) -> dict[str, str]:
    """
    Повертає {hash: full_url} для списку image_hash через /adimages endpoint.
    Запитує батчами по 50.
    """
    import json as _json
    result = {}
    for i in range(0, len(hashes), 50):
        batch = hashes[i:i + 50]
        data = _get(f"{base_url}/{acc_id}/adimages", {
            "access_token": token,
            "hashes": _json.dumps(batch),
            "fields": "id,name,url,width,height",
        })
        for img in data.get("data", []):
            # id має формат "act_123:hash"
            h = img["id"].split(":")[-1]
            result[h] = img.get("url", "")
    return result


# ─── Агрегація ───────────────────────────────────────────────────────────────

def _get_action(ins: dict, *types) -> float:
    for t in types:
        for a in ins.get("actions", []):
            if a["action_type"] == t:
                return float(a["value"])
    return 0.0


def _results_action_type_for_objective(objective: str) -> list[str]:
    """Повертає список action_type які Meta рахує як Results для даного objective."""
    return OBJECTIVE_RESULTS_ACTION.get(objective, [
        "lead", "offsite_conversion.fb_pixel_lead",
        "offsite_conversion.fb_pixel_custom", "onsite_conversion.lead_grouped",
    ])


def aggregate_by_name(ads: list[dict], campaign_meta: dict[str, dict]) -> dict[str, dict]:
    """Агрегує статистику по імені креативу. Збирає image_hash для отримання повного URL."""
    agg = defaultdict(lambda: {
        "label": "", "objective": "", "results_action_type": "",
        "thumbnail_url": "", "image_hash": "", "ad_ids": [],
        "spend": 0.0, "impressions": 0, "clicks": 0,
        "leads": 0.0, "custom_conv": 0.0, "landing_views": 0.0,
        "results": 0.0,
    })
    for ad in ads:
        name = ad["name"]
        cid  = ad.get("campaign_id", "")
        r    = agg[name]
        camp = campaign_meta.get(cid, {})
        r["label"]     = camp.get("label", "?")
        objective      = camp.get("objective", "")
        if not r["objective"]:
            r["objective"] = objective

        creative = ad.get("creative", {})
        thumb = creative.get("thumbnail_url", "")
        if thumb and not r["thumbnail_url"]:
            r["thumbnail_url"] = thumb

        # Беремо перший image_hash з asset_feed_spec
        if not r["image_hash"]:
            images = creative.get("asset_feed_spec", {}).get("images", [])
            if images:
                r["image_hash"] = images[0].get("hash", "")

        r["ad_ids"].append(ad["id"])

        ins = (ad.get("insights", {}).get("data") or [{}])[0]
        r["spend"]        += float(ins.get("spend", 0) or 0)
        r["impressions"]  += int(ins.get("impressions", 0) or 0)
        r["clicks"]       += int(ins.get("clicks", 0) or 0)
        r["leads"]        += _get_action(ins, "lead", "offsite_conversion.fb_pixel_lead")
        r["custom_conv"]  += _get_action(
            ins,
            "offsite_conversion.fb_pixel_custom",
            "onsite_conversion.lead_grouped",
        )
        r["landing_views"] += _get_action(ins, "landing_page_view")

        # Results — значення відповідно до objective кампанії
        result_types = _results_action_type_for_objective(objective)
        result_val   = _get_action(ins, *result_types)
        r["results"] += result_val
        # Запам'ятовуємо перший знайдений action_type як підпис
        if not r["results_action_type"] and result_val > 0:
            for at in result_types:
                if _get_action(ins, at) > 0:
                    r["results_action_type"] = at
                    break

    return dict(agg)


# ─── Images ──────────────────────────────────────────────────────────────────

def enrich_full_image_urls(
    agg: dict[str, dict],
    token: str,
    base_url: str,
    acc_id: str,
) -> None:
    """
    Для кожного креативу з image_hash підтягує повний URL через /adimages (in-place).
    Зберігає в r["full_image_url"]. Fallback — thumbnail_url.
    """
    hashes = list({r["image_hash"] for r in agg.values() if r.get("image_hash")})
    if not hashes:
        return
    print(f"  Отримую повні URL для {len(hashes)} image_hash...")
    hash_to_url = get_full_image_urls(token, base_url, acc_id, hashes)
    for r in agg.values():
        h = r.get("image_hash", "")
        r["full_image_url"] = hash_to_url.get(h, "") or r.get("thumbnail_url", "")
    found = sum(1 for r in agg.values() if r.get("full_image_url"))
    print(f"  Повних URL: {found}/{len(agg)}")


def download_images_b64(agg: dict[str, dict]) -> None:
    """
    Завантажує зображення як base64 (in-place).
    Пріоритет: full_image_url → thumbnail_url.
    Зберігає в r["thumb_b64"] (для UI) і r["image_b64"] (для Claude Vision).
    """
    total = len(agg)
    for i, (name, r) in enumerate(agg.items()):
        # Для Claude Vision — повне зображення
        if not r.get("image_b64"):
            url = r.get("full_image_url") or r.get("thumbnail_url", "")
            if url:
                try:
                    resp = requests.get(url, timeout=15)
                    if resp.status_code == 200:
                        ct  = resp.headers.get("content-type", "image/jpeg")
                        b64 = base64.b64encode(resp.content).decode()
                        r["image_b64"] = f"data:{ct};base64,{b64}"
                    else:
                        print(f"  Зображення {name[:40]}: HTTP {resp.status_code}")
                        r["image_b64"] = ""
                except Exception as exc:
                    print(f"  Зображення {name[:40]}: {exc}")
                    r["image_b64"] = ""

        # Для UI — thumbnail (менший розмір, швидше)
        if not r.get("thumb_b64"):
            url = r.get("thumbnail_url", "")
            if url:
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        ct  = resp.headers.get("content-type", "image/jpeg")
                        b64 = base64.b64encode(resp.content).decode()
                        r["thumb_b64"] = f"data:{ct};base64,{b64}"
                    else:
                        r["thumb_b64"] = ""
                except Exception:
                    r["thumb_b64"] = ""

        if (i + 1) % 25 == 0:
            print(f"  images: {i+1}/{total}")


# ─── Кеш ─────────────────────────────────────────────────────────────────────

def load_or_fetch(token: str, base_url: str, acc_id: str, force: bool = False) -> dict[str, dict]:
    """
    Якщо кеш є і force=False — повертає кешовані дані.
    Інакше — завантажує з API і кешує.
    """
    if CACHE_RAW.exists() and not force:
        print(f"Завантажую з кешу {CACHE_RAW}")
        with open(CACHE_RAW, "rb") as f:
            return pickle.load(f)

    print("Завантажую кампанії з Meta API...")
    campaign_meta = get_target_campaigns(token, base_url, acc_id)
    print(f"  Знайдено {len(campaign_meta)} цільових кампаній")

    print("Завантажую ads...")
    ads = get_all_ads(token, base_url, acc_id, set(campaign_meta.keys()))
    print(f"  Знайдено {len(ads)} ads")

    agg = aggregate_by_name(ads, campaign_meta)
    print(f"  Унікальних креативів: {len(agg)}")

    print("Отримую повні URL зображень...")
    enrich_full_image_urls(agg, token, base_url, acc_id)

    print("Завантажую зображення...")
    download_images_b64(agg)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_RAW, "wb") as f:
        pickle.dump(agg, f)
    print(f"Кеш збережено → {CACHE_RAW}")
    return agg

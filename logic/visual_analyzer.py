"""
Візуальний аналіз рекламних креативів через ENOT/Claude Vision.
9 атрибутів через Claude + bg_type програмно через HSV-аналіз пікселів.
"""
import json
import base64
import pickle
import time
import colorsys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_VISUAL = CACHE_DIR / "visual_attrs.pkl"

VISUAL_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "mascot": {"type": "boolean"},
        "screenshot": {"type": "boolean"},
        "cta_button": {"type": "boolean"},
        "offer_type": {"type": "string", "enum": ["rate", "fx", "payroll", "cashback", "review", "payments", "kep", "acquiring", "general", "other"]},
        "has_number": {"type": "boolean"},
        "style":  {"type": "string", "enum": ["minimal", "illustrated", "product", "card", "mixed", "video"]},
        "text_lines": {"type": "integer"},
        "ui_elements": {"type": "boolean"},
        "specific_benefit": {"type": "boolean"},
    },
    "required": ["mascot", "screenshot", "cta_button", "offer_type",
                 "has_number", "style", "text_lines", "ui_elements", "specific_benefit"],
})

_SYSTEM_PROMPT = (
    "You are an expert at analyzing Ukrainian B2B bank advertising creatives for monobank Business. "
    "Analyze the provided ad image and extract visual attributes. "
    "Context: Facebook/Instagram ads for monobank Business (B2B banking). "
    "monobank has a black cat mascot. Products: FOP (entrepreneurs), YO (legal entities), Acquiring (terminals). "
    "Answer ONLY based on what you literally see in the image."
)

_USER_PROMPT = (
    "Analyze this ad creative and return 9 attributes:\n"
    "1. mascot: cat mascot visible? true/false\n"
    "2. screenshot: mobile app screenshot visible? true/false\n"
    "3. cta_button: CTA button ON THE IMAGE ITSELF? true/false\n"
    "4. offer_type: 'rate' (interest %), 'fx' (currency/SWIFT), 'payroll' (salary project), 'cashback', 'review' (testimonial), 'payments' (no-fee), 'kep' (e-signature), 'acquiring' (terminals/QR), 'general'\n"
    "5. has_number: specific number (14%, 0%, 1%) in main text? true/false\n"
    "6. style: 'minimal' (text on solid bg), 'illustrated' (mascot scene), 'product' (app screenshots as main), 'card' (testimonial card), 'mixed' (text+UI), 'video' (video frame)\n"
    "7. text_lines: count of separate text blocks (integer 1–8)\n"
    "8. ui_elements: app UI pills/cards/buttons visible? true/false\n"
    "9. specific_benefit: promises SPECIFIC benefit (0% fee, 14% annual) vs generic? true/false"
)


# ─── Програмне визначення фону ───────────────────────────────────────────────

def _rgb_to_bg_type(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360

    if v < 0.18:
        return "black"
    if v > 0.88 and s < 0.12:
        return "white"
    if v > 0.65 and s < 0.12:
        return "light_grey"
    if v < 0.42 and s < 0.20:
        return "chalk"
    if 190 <= h_deg <= 270:
        if v < 0.25:
            return "navy"
        if v < 0.55:
            return "dark_blue"
        return "light_blue"
    if 170 <= h_deg < 190 and s > 0.3:
        return "light_blue"
    if 270 <= h_deg <= 320 and s > 0.25:
        return "purple"
    if v < 0.45:
        return "dark_gradient"
    return "other"


def detect_bg_type(img_bytes: bytes) -> str:
    """Визначає тип фону по пікселях країв зображення."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        pixels = []
        for x in range(0, w, max(1, w // 20)):
            pixels.append(img.getpixel((x, 0)))
            pixels.append(img.getpixel((x, h - 1)))
        for y in range(0, h, max(1, h // 20)):
            pixels.append(img.getpixel((0, y)))
            pixels.append(img.getpixel((w - 1, y)))
        for cx in [0, w // 4, w * 3 // 4, w - 1]:
            for cy in [0, h // 4, h * 3 // 4, h - 1]:
                pixels.append(img.getpixel((cx, cy)))
        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)
        return _rgb_to_bg_type(avg_r, avg_g, avg_b)
    except Exception:
        return "other"


def _analyze_one(name: str, image_b64: str, thumbnail_url: str, enot_key: str, enot_url: str) -> tuple[str, dict | None, str]:
    """Аналізує один креатив. Повертає (name, attrs, status)."""

    # Пріоритет: вже завантажений image_b64, інакше — завантажуємо thumbnail
    b64_data = image_b64
    if not b64_data and thumbnail_url:
        try:
            resp = requests.get(thumbnail_url, timeout=12)
            if resp.status_code == 200:
                ct_ = resp.headers.get("content-type", "image/jpeg")
                b64_data = f"data:{ct_};base64," + base64.standard_b64encode(resp.content).decode()
        except Exception:
            pass


    if not b64_data:
        return name, None, "no_image"

    # Витягуємо content-type і байти з data URI
    if b64_data.startswith("data:"):
        header, raw_b64 = b64_data.split(",", 1)
        ct = header.split(":")[1].split(";")[0]
    else:
        raw_b64 = b64_data
        ct = "image/jpeg"

    img_bytes = base64.b64decode(raw_b64)

    # Автодетектуємо реальний тип по magic bytes (Meta іноді бреше content-type)
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ct = "image/png"
    elif img_bytes[:3] == b"\xff\xd8\xff":
        ct = "image/jpeg"
    elif img_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ct = "image/gif"
    elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        ct = "image/webp"

    ext = ct.split("/")[-1].replace("jpeg", "jpg")
    api_url = enot_url.rstrip("/").replace("/3aw", "/3a")

    for attempt in range(3):
        try:
            api_resp = requests.post(
                api_url,
                headers={"X-API-Key": enot_key},
                data={
                    "prompt":     _SYSTEM_PROMPT,
                    "user_input": _USER_PROMPT,
                    "schema":     VISUAL_SCHEMA,
                    "model":      "claude-3.7",
                },
                files={"file": (f"{name[:40]}.{ext}", img_bytes, ct)},
                timeout=60,
            )

            if api_resp.status_code == 200:
                result = api_resp.json().get("result", {})
                result["bg_type"] = detect_bg_type(img_bytes)
                return name, result, "ok"
            elif api_resp.status_code in (502, 503):
                wait = 4 * (attempt + 1)
                print(f"  {api_resp.status_code} retry {attempt+1}/3 для '{name[:30]}', чекаю {wait}с")
                time.sleep(wait)
            else:
                return name, None, f"api_{api_resp.status_code}"
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                return name, None, f"error_{e}"

    return name, None, "max_retries"


def load_cache() -> dict[str, dict]:
    if CACHE_VISUAL.exists():
        with open(CACHE_VISUAL, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(visual: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_VISUAL, "wb") as f:
        pickle.dump(visual, f)


def analyze_all(
    agg: dict[str, dict],
    enot_key: str,
    enot_url: str,
    workers: int = 3,
    force: bool = False,
) -> dict[str, dict]:
    """
    Аналізує всі креативи в agg.
    Пропускає вже проаналізовані (з кешу).
    Зберігає прогрес у кеш кожні 10 записів.
    """
    visual = {} if force else load_cache()
    missing = [
        (name, r.get("image_b64", ""), r.get("thumbnail_url", ""))
        for name, r in agg.items()
        if name not in visual
    ]

    if not missing:
        print(f"Всі {len(visual)} креативів вже проаналізовано (з кешу).")
        return visual

    print(f"Потрібно проаналізувати: {len(missing)} (пропущено {len(agg)-len(missing)} з кешу)")

    done = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_one, name, img_b64, thumb_url, enot_key, enot_url): name
            for name, img_b64, thumb_url in missing
        }
        for future in as_completed(futures):
            name, result, status = future.result()
            done += 1
            if result:
                visual[name] = result
            else:
                errors.append(f"{name}: {status}")

            if done % 10 == 0:
                save_cache(visual)
                print(f"  {done}/{len(missing)} — ОК: {len(visual)}, помилок: {len(errors)}")
            time.sleep(0.4)

    save_cache(visual)
    print(f"\nВізуальний аналіз завершено: {len(visual)} ОК, {len(errors)} помилок")
    if errors:
        print("Помилки:", errors[:5])
    return visual

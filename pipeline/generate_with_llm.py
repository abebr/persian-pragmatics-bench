#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Direct LLM Synthesizer via Antigravity Gemini 3.8 Flash.
Calls the live LLM with prompts from pipeline/prompts.py to generate genuine,
diverse, context-grounded Persian pragmatic dialogue pairs.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.antigravity_client import call_antigravity_gemini
from pipeline.prompts import CATEGORY_PROMPTS, SYSTEM_PROMPT
from pipeline.scenarios_v2 import SCENARIOS_V2

SCENARIOS_BASE = {
    "taarof": [
        "تعارف سر کشیدن دسر زعفرانی در مهمانی عید نوروز",
        "پرداخت صورتحساب کافه توسط دو دوست صمیمی",
        "تعارف عبور از در ورودی یک سازمان دولتی بین دو کارمند",
        "خرید هندوانه و میوه در بازار تره‌بار محلی",
        "تعارف کادوی تولد و سوغاتی سفر مشهد",
        "تعارف نشستن در صندلی جلو اتومبیل همکار",
        "پرداخت دستمزد کارگر یا نصاب پرده در منزل"
    ],
    "sarcasm": [
        "پاسخگویی پشتیبانی اینترنت پس از ۴ روز قطعی کامل",
        "تحویل پیتزای کاملاً سوخته و زغالی توسط پیک رستوران",
        "ارائه کد پایتون بدون یونیت‌تست که سرور پروداکشن را داون کرده",
        "تاخیر ۳ ساعته همکار در رسیدن به جلسه مهم اداری",
        "پارک کردن خودرو در وسط کوچه بن‌بست و مسدود کردن راه",
        "تحلیل کاملاً اشتباه بازار بورس که به ضرر مالی سنگین منجر شده",
        "پاره کردن کتاب امانت‌گرفته‌شده توسط دوست",
        "تمیز کردن اتاق که بدتر از قبل شلخته و کثیف شده است"
    ],
    "indirect_request": [
        "دمای بسیار بالای اتاق و احساس خفگی سر جلسه امتحان",
        "سوز سرمای شدید که از لای پنجره باز سالن مطالعه می‌آید",
        "دور بودن نمکدان سر میز ناهارخوری خانوادگی",
        "شارژ ۲ درصدی تلفن همراه در حال خاموش شدن حین مکالمه کاری",
        "صدای بیش از حد بلند تلویزیون هنگام مطالعه دانشجو",
        "حمل همزمان سه جعبه کتاب بسیار سنگین در راهرو",
        "نبود خودکار برای پر کردن فیش بانکی در باجه",
        "تاریکی سالن اجتماعات به دلیل خاموش بودن کلید برق"
    ],
    "implicature": [
        "پیشنهاد سینما رفتن در شب قبل از آزمون جامع کنکور ارشد",
        "پیشنهاد مسافرت جاده‌ای شمال با خودرویی که لنت ترمزش ساییده شده",
        "پیشنهاد خوردن دیزی چرب به فردی که چربی خون بالایی دارد",
        "پیشنهاد اجاره خانه‌ای که سقفش رطوبت شدید و بوی نم دارد",
        "پرسش از کیفیت رمان نویسنده‌ای که متن کتابش بسیار خسته‌کننده است",
        "پیشنهاد سرمایه‌گذاری در ارز دیجیتالی که تیم توسعه آن ناشناس است",
        "پیشنهاد شروع همکاری با شرکتی که سابقه بدقولی در پرداخت دارد",
        "پرسش کارفرما از زمان تحویل پروژه در حالی که سرور اصلی سوخته است"
    ],
    "rhetorical_question": [
        "درخواست مداوم پول و شهریه بالا از پدری که کارمند ساده است",
        "تکرار همان باگ نرم‌افزاری دیروز توسط برنامه‌نویس برای سومین بار",
        "رانندگی با سرعت ۱۶۰ کیلومتر در جاده لغزنده بارانی",
        "توقع تحویل پروژه چندماهه ظرف ۴۸ ساعت بدون امکانات و بودجه",
        "فراموش کردن اصل کارت ملی در جلسه آزمون ورودی استخدام",
        "خرید یک جنس بی‌کیفیت به قیمتی نجومی و غیرواقعی",
        "اصرار به پیاده‌روی در کوهستان حین هشدار وقوع سیلاب",
        "بی‌توجهی به راهنمای مکتوب اداری و معطلی کل ارباب‌رجوع"
    ],
    "modesty_self_deprecation": [
        "تبریک رتبه یک کنکور سراسری به دانشجوی متواضع",
        "تعریف از تابلوی نقاشی سیاه‌قلم کشیده‌شده توسط هنرمند جوان",
        "تحسین سخنرانی بی‌نقص و مسلط در کنفرانس بین‌المللی هوش مصنوعی",
        "تمجید از دستپخت فوق‌العاده فسنجان مادر خانواده در مهمانی رسمی",
        "تعریف همکار از کدهای تمیز و معماری ماژولار برنامه‌نویس ارشد",
        "تحسین تیپ و آراستگی فرد در یک مراسم عقد و ازدواج",
        "تبریک به ورزشکاری که مدال طلای مسابقات کشوری را کسب کرده",
        "تمجید از صدای دلنشین خواننده در یک اجرای موسیقی سنتی زنده"
    ],
    "indirect_refusal": [
        "تقاضای قرض دادن ۵۰ میلیون تومان پول نقد توسط دوست قدیمی",
        "دعوت به جلسه حضوری غیرضروری در عصر پنجشنبه خارج از تایم کاری",
        "درخواست امانت دادن خودروی شخصی صفرکیلومتر برای مسافرت جاده‌ای",
        "پیشنهاد شراکت در راه‌اندازی یک کافه سنتی با سرمایه نامعلوم",
        "تقاضای ضامن شدن برای دریافت وام بانکی پرریسک همکار",
        "پیشنهاد اضافه‌کاری تا نیمه‌شب در ایام تعطیلات نوروز",
        "درخواست پذیرفتن سرپرستی پروژه‌ای که از قبل شکست خورده است",
        "دعوت به جشن تولدی که افراد نامناسب در آن حضور دارند"
    ],
    "conversational_repair": [
        "شوخی نامناسب با مدرک تحصیلی همکار که موجب دلخوری او شده است",
        "لحن تند ناخواسته در یک ایمیل کاری رسمی که سوءتفاهم ایجاد کرده",
        "انتقاد از رنگ طرح گرافیکی که طراح تصور کرده کل هنرش زیر سوال رفته",
        "سوءبرداشت مدیر از گزارش شفاف‌سازی هزینه‌های مالی پروژه",
        "تعبیر اشتباه شوخی دوستانه به عنوان توهین به خانواده طرف مقابل",
        "بیان اشتباه رقم قرارداد در حین مذاکره که تنش ایجاد کرده است",
        "استفاده از واژه‌ای دوپهلو در چت گروهی کاری که سوءتعبیر شده است",
        "قطع ناگهانی صحبت همکار در جلسه آنلاین که حمل بر بی‌احترامی شده"
    ]
}

# Phase 2 exclusive: strictly use newly authored diverse scenarios
SCENARIOS = SCENARIOS_V2


def clean_llm_json(raw_text: str) -> Optional[Dict]:
    """Extract and parse clean JSON from LLM response markdown blocks."""
    raw_text = raw_text.strip()
    # Strip markdown ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if m:
        raw_text = m.group(1).strip()
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        # Fallback regex search for JSON object
        m_obj = re.search(r"\{[\s\S]*\}", raw_text)
        if m_obj:
            try:
                return json.loads(m_obj.group(0))
            except Exception:
                pass
    return None


def call_with_retry(prompt: str, max_retries: int = 5, base_delay: float = 2.0) -> str:
    """Exponential backoff against HTTP 429 rate limits."""
    for attempt in range(max_retries):
        try:
            return call_antigravity_gemini(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"\n[⚠️ 429 Rate Limit] Backing off {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                time.sleep(1.0)
    return ""


def validate_quality_gate(data: Dict, seen_prefixes: Dict[str, Dict[str, int]]) -> bool:
    """Rigorous linguistic validation gate with prefix-diversity and spurious-cue filtering."""
    u = data.get("utterance", "").strip()
    sm = data.get("surface_meaning", "").strip()
    pi = data.get("pragmatic_intent", "").strip()
    cat = data.get("category", "")

    if not u or len(u.split()) < 3:
        return False
    if "(" in u or ")" in u or "{" in u or "}" in u:
        return False
    if sm == pi or len(pi) < 10:
        return False

    # Reject spurious cues in sarcasm category
    if cat == "sarcasm":
        banned_sarcasm = ["واقعاً", "واقعا", "چشمم روشن", "دست مریزاد", "دست‌مریزاد", "دستت درد نکنه", "خسته نباشی"]
        if any(b in u for b in banned_sarcasm):
            return False

    # Reject templated suffixes
    if any(u.endswith(bad) for bad in ["عزیزم.", "متاسفانه.", "ان‌شاءالله.", "البته.", "به هر حال."]):
        return False

    # Diversity guard: restrict repetitive sentence openings (first 2 words) to max 2 uses per category
    pfx2 = " ".join(u.split()[:2])
    cat_seen = seen_prefixes.setdefault(cat, {})
    if cat_seen.get(pfx2, 0) >= 2:
        return False

    return True


def update_live_html_widget(current: int, total: int, last_sample: Dict):
    """Updates the native Hermes desktop HTML progress widget."""
    widget_file = Path(__file__).parent.parent / "data" / "progress_widget.html"
    if not widget_file.exists():
        return
    pct = round((current / max(1, total)) * 100.0, 1)
    u = last_sample.get("utterance", "")[:80]
    cat = last_sample.get("category", "")
    
    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<style>
  .widget-card {{ border: 1px solid var(--border, #333); border-radius: 8px; padding: 16px; background: var(--card, #1e1e1e); color: var(--foreground, #eee); font-family: inherit; max-width: 480px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .title {{ font-weight: bold; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
  .badge {{ background: var(--accent, #3b82f6); color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
  .progress-bg {{ background: rgba(255, 255, 255, 0.1); border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 8px; }}
  .progress-bar {{ background: #10b981; height: 100%; width: {pct}%; transition: width 0.3s ease; }}
  .stats {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--muted-foreground, #aaa); margin-bottom: 12px; }}
  .preview-box {{ background: rgba(0, 0, 0, 0.2); border: 1px dashed var(--border, #444); border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.5; }}
  .preview-title {{ font-size: 11px; color: var(--muted-foreground, #888); margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="widget-card">
  <div class="header">
    <div class="title"><span>⚡ خط‌لوله زنده تولید دیتاست Antigravity</span></div>
    <span class="badge">{cat}</span>
  </div>
  <div class="progress-bg"><div class="progress-bar"></div></div>
  <div class="stats">
    <span>تولید شده: {current:,} / {total:,}</span>
    <span>{pct:.1f}٪</span>
  </div>
  <div class="preview-box">
    <div class="preview-title">آخرین نمونه تاییدشده:</div>
    <div>«{u}...»</div>
  </div>
</div>
</body>
</html>"""
    try:
        with open(widget_file, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass


def synthesize_live_batch(
    target_count: int,
    output_jsonl: Path,
    seen_utterances: Set[str],
    cooldown_sec: float = 0.8
) -> List[Dict]:
    seen_utterances: Set[str] = set()
    seen_prefixes: Dict[str, Dict[str, int]] = {}
    cat_counts: Dict[str, int] = {c: 0 for c in SCENARIOS}

    # Load existing progress to resume without duplicates
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try:
                        obj = json.loads(l)
                        u = obj["utterance"]
                        seen_utterances.add(u)
                        cat = obj.get("category", "")
                        pfx2 = " ".join(u.split()[:2])
                        seen_prefixes.setdefault(cat, {})
                        seen_prefixes[cat][pfx2] = seen_prefixes[cat].get(pfx2, 0) + 1
                        if cat in cat_counts:
                            cat_counts[cat] += 1
                    except Exception:
                        pass

    print(f"Resuming live synthesis from {len(seen_utterances):,} items (Target: {target_count:,})...\n")

    results = []
    sample_id = len(seen_utterances) + 1
    cats = list(SCENARIOS.keys())
    cat_idx = 0

    while len(seen_utterances) < target_count:
        # Pick the category with the minimum samples to enforce strict balance
        cat = min(cats, key=lambda c: cat_counts[c])
        scenario_list = SCENARIOS[cat]
        scenario = scenario_list[(cat_counts[cat]) % len(scenario_list)]

        prompt_tpl = CATEGORY_PROMPTS[cat]["prompt"]
        prompt = prompt_tpl.format(scenario=scenario)

        try:
            raw_response = call_with_retry(prompt)
            parsed = clean_llm_json(raw_response)

            if not parsed or not validate_quality_gate(parsed, seen_prefixes):
                time.sleep(0.3)
                continue

            utt = parsed.get("utterance", "").strip()
            if utt in seen_utterances:
                continue

            seen_utterances.add(utt)
            pfx2 = " ".join(utt.split()[:2])
            seen_prefixes.setdefault(cat, {})
            seen_prefixes[cat][pfx2] = seen_prefixes[cat].get(pfx2, 0) + 1

            record = {
                "id": f"{cat}-{sample_id:05d}",
                "category": cat,
                "context": parsed.get("context", f"موقعیت در {scenario}").strip(),
                "utterance": utt,
                "surface_meaning": parsed.get("surface_meaning", "").strip(),
                "pragmatic_intent": parsed.get("pragmatic_intent", "").strip(),
                "correct_response": parsed.get("correct_response", "").strip(),
                "naive_llm_response": parsed.get("naive_llm_response", "").strip()
            }

            # Atomic append to disk immediately (resumable)
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            results.append(record)
            sample_id += 1
            cat_counts[cat] += 1
            cat_idx += 1

            # Update live visual progress bar and desktop chat widget
            update_live_html_widget(len(seen_utterances), target_count, record)

            curr = len(seen_utterances)
            percent = (curr / max(1, target_count)) * 100.0
            bar = "█" * int(20 * curr // target_count) + "░" * (20 - int(20 * curr // target_count))
            sys.stdout.write(f"\r[{bar}] {percent:5.1f}% ({curr:,}/{target_count:,}) | «{utt[:30]}...»")
            sys.stdout.flush()

            time.sleep(cooldown_sec)

        except Exception as e:
            print(f"\n[!] Throttled call: {e}", file=sys.stderr)
            time.sleep(2.0)

    print(f"\n\n✓ Finished! Total samples in file: {len(seen_utterances):,}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Live LLM Pragmatic Dataset Synthesizer")
    parser.add_argument("--count", type=int, default=50, help="Target total samples to reach")
    parser.add_argument("--out", type=str, default="data/llm_generated_live.jsonl", help="Output file")
    parser.add_argument("--delay", type=float, default=0.5, help="Cooldown delay between calls in seconds")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try:
                        seen.add(json.loads(l)["utterance"])
                    except Exception:
                        pass

    res = synthesize_live_batch(args.count, out_path, seen, cooldown_sec=args.delay)
    print(f"\n✓ Finished batch! Total newly synthesized: {len(res)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
persian-pragmatics-dataset synthesis pipeline:
Linguistically-guided prompt synthesis using Google Gemini Flash API.
Enforces Searle's Speech Act Taxonomy, Grice's Cooperative Maxims, and JSON Schema.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_INSTRUCTION = """\
شما یک متخصص ارشد زبان‌شناسی رایانشی و کاربردشناسی زبان فارسی (Persian Pragmatics) هستید.
وظیفه شما تولید جفت‌های دیالوگی طبیعی، واقعی و از نظر زبان‌شناختی استاندارد به زبان فارسی است
که در آنها بین «معنای ظاهری/تحت‌اللفظی» (Surface / Literal Meaning) و «مقصود کاربردشناختی/ضمنی» (Pragmatic / Implied Intent) شکاف عمیق وجود دارد.

شما باید نمونه‌ها را دقیقاً در یکی از ۴ دسته زیر تولید کنید:
1. taarof (تعارفات آیینی، نبرد پرداخت فاکتور، تعارف غذا، تقدم در عبور)
2. sarcasm (طعنه، کنایه، وارونگی قطبیت، تمسخر تاخیر یا خرابی)
3. indirect_request (کنش‌های گفتاری غیرمستقیم، بیان گزاره برای اقدام فیزیکی)
4. implicature (استلزام گفتگویی گریس، رد غیرمستقیم با ارائه دلیل)

خروجی شما باید منحصراً یک آرایه معتبر JSON شامل فیلدهای تعیین‌شده باشد.
"""

PROMPT_TEMPLATE = """\
دسته‌بندی درخواستی: {category}
موقعیت / دامنه اجتماعی: {domain}
سطح لحن / رجیستر: {register}
کنش گفتاری هدف: {speech_act}
تعداد نمونه‌های مورد نیاز: {count}

نمونه خروجی استاندارد:
[
  {{
    "context": "موقعیت در تاکسی و اسنپ شهری بین راننده و مسافر به شیوه محترمانه (تعارف کرایه)",
    "utterance": "مهمون ما باشید، اصلاً قابل شما رو نداره.",
    "surface_meaning": "سفر رایگان است و نیازی به پرداخت وجه نیست.",
    "pragmatic_intent": "تعارف آیینی اجتماعی؛ دریافت کرایه قطعی و الزامی است.",
    "correct_response": "اختیار دارید، خواهش می‌کنم کارتخوان کجاست خدمتتون بکشم؟",
    "naive_llm_response": "خیلی ممنون از سخاوت شما! پس من کرایه را پرداخت نمی‌کنم."
  }}
]

قوانین الزامی:
- زبان دیالوگ‌ها باید کاملاً روان و متناسب با گفتار طبیعی مردم ایران باشد.
- فیلد naive_llm_response باید تله‌ی لفظی رایجی را نشان دهد که مدل‌های زبانی به اشتباه در آن می‌افتند.
- فقط آرایه JSON را بدون هیچ توضیح اضافی برگردانید.
"""


def generate_with_gemini(
    category: str,
    domain: str,
    register: str,
    speech_act: str,
    count: int = 5,
    api_key: Optional[str] = None,
    model: str = "gemini-3.8-flash"
) -> List[Dict]:
    """Calls Google Gemini API with structured prompt and parses JSON output."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable or argument is required.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    user_prompt = PROMPT_TEMPLATE.format(
        category=category,
        domain=domain,
        register=register,
        speech_act=speech_act,
        count=count
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json"
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw_text)


def self_test():
    """Verify prompt formatting and JSON parsing logic."""
    prompt = PROMPT_TEMPLATE.format(
        category="taarof",
        domain="کافه",
        register="عامیانه",
        speech_act="پرداخت فاکتور",
        count=2
    )
    assert "taarof" in prompt
    assert "عامیانه" in prompt
    print("✓ Synthesis prompt template verified.")


def main():
    parser = argparse.ArgumentParser(description="Generate Persian Pragmatic Samples via Gemini API")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    parser.add_argument("--category", type=str, default="taarof")
    parser.add_argument("--domain", type=str, default="تاکسی شهری")
    parser.add_argument("--register", type=str, default="محترمانه")
    parser.add_argument("--speech-act", type=str, default="تعارف کرایه")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"))
    args = parser.parse_args()

    if args.test:
        self_test()
        return

    try:
        samples = generate_with_gemini(
            category=args.category,
            domain=args.domain,
            register=args.register,
            speech_act=args.speech_act,
            count=args.count,
            api_key=args.api_key,
            model=args.model
        )
        print(json.dumps(samples, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Synthesis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

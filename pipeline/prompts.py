#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Standardized Prompt Registry & Directives.
Houses all verified linguistic prompts and constraints used for synthesizing
and auditing data via Gemini 3.8 Flash. Serves as Appendix A in the academic paper.
"""

from pathlib import Path
from typing import Dict, List

# System prompt defining the expert role and linguistic boundaries
SYSTEM_PROMPT = """\
شما یک استاد و پژوهشگر ارشد زبان‌شناسی رایانشی (Computational Linguistics) متخصص در کاربردشناسی زبان فارسی (Persian Pragmatics) هستید.
وظیفه شما تولید نمونه‌های دیالوگی طبیعی، ظریف، واقعی و عاری از هرگونه ترجمه ماشینی یا ساختارهای قالبی و کلیشه‌ای است.
تمرکز بر موقعیت‌هایی است که در آنها شکاف عمیق، متناقض یا چندلایه میان «معنای وضعی/تحت‌اللفظی» (Surface Meaning) و «نیروی منظورشناختی/قصد گوینده» (Illocutionary Force / Pragmatic Intent) وجود دارد.
"""

# Specialized task-specific prompts across the 8 categories
CATEGORY_PROMPTS: Dict[str, Dict[str, str]] = {
    "taarof": {
        "title": "تعارفات آیینی و ادب اجتماعی (Ritual Politeness & Ta'arof)",
        "prompt": """\
دسته: تعارفات آیینی (taarof)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده بر اساس سنت تعارف، گزاره‌ای را مطرح می‌کند که معنای ظاهری آن رایگان بودن، پیشکش کردن یا اصرار بر تعارف است، اما طبق عرف فرهنگی جامعه ایران، دریافت وجه، امتناع از تصاحب یا انجام وظیفه الزامی و حتمی است.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح کامل بافت و رابطه اجتماعی بین دو طرف",
  "utterance": "جمله طبیعی، محاوره‌ای و اصیل گوینده بدون هیچ پرانتز یا علامت اضافی",
  "surface_meaning": "معنای لغوی و تحت‌اللفظی جمله",
  "pragmatic_intent": "مقصود واقعی، هنجار اجتماعی و الزام ضمنی گوینده",
  "correct_response": "پاسخ صحیح، طبیعی و آگاه به فرهنگ از سوی یک ایرانی",
  "naive_llm_response": "پاسخ تحت‌اللفظی، مکانیکی و اشتباهی که یک چت‌بات ساده لوح می‌دهد"
}}
"""
    },
    "sarcasm": {
        "title": "طعنه، کنایه و وارونگی قطبیت (Sarcasm & Irony)",
        "prompt": """\
دسته: طعنه و کنایه (sarcasm)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده با استفاده از کلمات ستایشی، اغراق مثبت یا تحسین، در حقیقت در حال تمسخر، توبیخ و ابراز نارضایتی شدید از تاخیر، خرابی یا اشتباه طرف مقابل است (وارونگی کامل قطبیت معنایی).

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح موقعیت، علت خشم گوینده و خرابکاری رخ‌داده",
  "utterance": "جمله کنایه‌آمیز طبیعی گوینده بدون هیچ پرانتز یا علامت اضافی",
  "surface_meaning": "تفسیر تحت‌اللفظی مبتنی بر تمجید و رضایت",
  "pragmatic_intent": "نارضایتی شدید، تمسخر و اعتراض واقعی گوینده",
  "correct_response": "عذرخواهی، درک کنایه و پذیرش مسئولیت خطا",
  "naive_llm_response": "تشکر از تمجید گوینده و به دام افتادن در ظاهر مثبت کلمات"
}}
"""
    },
    "indirect_request": {
        "title": "کنش‌های گفتاری غیرمستقیم (Searle's Indirect Speech Acts)",
        "prompt": """\
دسته: درخواست غیرمستقیم (indirect_request)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده بدون بیان فعل امری یا تقاضای مستقیم، با بیان یک گزاره اخباری درباره شرایط محیطی (دما، نور، صدا) یا وضعیت جسمی/ابزاری خود، از مخاطب درخواست انجام یک عمل اصلاحی فوری دارد.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح بافت فیزیکی، رابطه طرفین و مانع ایجاد شده",
  "utterance": "گزاره اخباری گوینده که حامل درخواست عمل است (بدون پرانتز)",
  "surface_meaning": "صرفاً یک گزارش علمی یا مشاهده توصیفی از محیط",
  "pragmatic_intent": "درخواست انجام یک عمل ملموس فیزیکی از سوی مخاطب",
  "correct_response": "اقدام عملی فوری در پاسخ به نیاز ضمنی گوینده",
  "naive_llm_response": "پاسخ تئوریک یا تایید گزارش فیزیکی بدون انجام عمل"
}}
"""
    },
    "implicature": {
        "title": "استلزام گفتگویی گریس (Gricean Conversational Implicature)",
        "prompt": """\
دسته: استلزام گفتگویی (implicature)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده در پاسخ به یک پیشنهاد، دعوت یا سوال دوگزینه‌ای، به جای بله/خیر صریح، با نقض اصل ربط (Maxim of Relation) و بیان یک واقعیت بیرونی (مانند بیماری، امتحان، مشکل فنی)، به صورت غیرمستقیم پیام منفی یا مخالفت خود را می‌رساند.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح پیشنهاد مطرح‌شده و رابطه گوینده و مخاطب",
  "utterance": "پاسخ ضمنی گوینده شامل مانع یا توضیح (بدون پرانتز)",
  "surface_meaning": "ارائه اطلاعات تقویمی، فنی یا بیولوژیکی مستقل",
  "pragmatic_intent": "رد قاطع و مودبانه پیشنهاد بر پایه استنتاج منطقی",
  "correct_response": "درک مانع و لغو یا به تعویق انداختن موضوع",
  "naive_llm_response": "اصرار بر دریافت پاسخ بله/خیر صریح به دلیل نفهمیدن ربط جمله"
}}
"""
    },
    "rhetorical_question": {
        "title": "پرسش‌های بلاغی و توبیخی (Rhetorical Questions)",
        "prompt": """\
دسته: پرسش بلاغی (rhetorical_question)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده با طرح یک پرسش استفهامی انکاری یا مبالغه‌آمیز (نظیر 'مگه سر گنج نشستم؟' یا 'مگه از پشت کوه اومدم؟') در حقیقت قصد توبیخ، رد تقاضا و تاکید بر غیرمنطقی بودن رفتار طرف را دارد و به هیچ عنوان منتظر پاسخ پرسش نیست.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح خواسته غیرمنطقی و موقعیت توبیخ",
  "utterance": "پرسش بلاغی و انکاری گوینده (بدون پرانتز)",
  "surface_meaning": "پرسش در مورد اطلاعات واقعی جغرافیایی، فیزیولوژیک یا مالی",
  "pragmatic_intent": "رد قاطع، سرزنش شدید و تبیین نامعقول بودن خواسته",
  "correct_response": "عذرخواهی، درک غیرمنطقی بودن موضوع و تعدیل رفتار",
  "naive_llm_response": "پاسخ علمی به ظاهر پرسش (مثلاً پاسخ به وجود فیزیکی گنج یا گیاه)"
}}
"""
    },
    "modesty_self_deprecation": {
        "title": "فروتنی و شکسته‌نفسی آیینی (Modesty & Self-Deprecation)",
        "prompt": """\
دسته: فروتنی و شکسته‌نفسی (modesty_self_deprecation)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده در پاسخ به تحسین، تمجید یا تبریک دیگران، با ادعای بی‌ارزش بودن کار خود، نسبت دادن موفقیت به شانس یا ابراز شاگردی، بر اساس هنجار فرهنگی شکسته‌نفسی و حفظ آبرو تواضع می‌کند.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح موفقیت فرد و تمجیدی که از او صورت گرفته",
  "utterance": "جمله متواضعانه و شکسته‌نفسی گوینده (بدون پرانتز)",
  "surface_meaning": "اعتراف به بی‌ارزشی کار، شانس مطلق بودن یا بی‌هنری خود",
  "pragmatic_intent": "رعایت ادب و تواضع فرهنگی در برابر تحسین جامعه",
  "correct_response": "رد شکسته‌نفسی و بازتایید شایستگی و هنر فرد",
  "naive_llm_response": "تایید بی‌ارزش بودن کار و توصیه به آموزش و تلاش بیشتر"
}}
"""
    },
    "indirect_refusal": {
        "title": "رد غیرمستقیم تعهد و خواهش (Indirect Refusal)",
        "prompt": """\
دسته: رد غیرمستقیم (indirect_refusal)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده برای امتناع از پذیرش یک مسئولیت، درخواست مالی یا تعهد کاری، به جای پاسخ رد صریح و تهاجمی، با موکول کردن به مشورت با همسر/شریک یا اشاره به ضیق وقت، محترمانه 'نه' می‌گوید.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح تقاضای مطرح‌شده و محذوریت گوینده",
  "utterance": "پاسخ دیپلماتیک و رد غیرمستقیم گوینده (بدون پرانتز)",
  "surface_meaning": "وعده تصمیم‌گیری آتی پس از مشورت یا رفع مانع زمانی",
  "pragmatic_intent": "امتناع قطعی و محترمانه از پذیرش تقاضا بدون دلخوری شخصی",
  "correct_response": "درک معذوریت گوینده و عدم اصرار بیشتر",
  "naive_llm_response": "انتظار خوش‌بینانه برای اتمام مشورت و دریافت پاسخ مثبت"
}}
"""
    },
    "conversational_repair": {
        "title": "ترمیم مکالمه و رفع سوءتفاهم (Conversational Repair)",
        "prompt": """\
دسته: ترمیم مکالمه (conversational_repair)
بافت و سناریو: {scenario}
دستورالعمل تولید:
یک نمونه دیالوگ فارسی تولید کنید که در آن گوینده پس از متوجه شدن سوءبرداشت یا دلخوری مخاطب از یک شوخی یا لحن تند، اقدام به بازسازی رابطه کلامی و توضیح نیت واقعی خود می‌کند تا تنش روانی مکالمه را خنثی سازد.

خروجی باید منحصراً در قالب یک شیء JSON با ساختار زیر باشد:
{{
  "context": "شرح سوءتفاهم پیش‌آمده در جلسه یا گفتگو",
  "utterance": "جمله ترمیمی و شفاف‌ساز گوینده (بدون پرانتز)",
  "surface_meaning": "نقض سخن قبلی و اظهار ضعف در رساندن کلام",
  "pragmatic_intent": "دلجویی، رفع سوءتعبیر و بازگرداندن اعتماد به مکالمه",
  "correct_response": "پذیرش عذرخواهی و تایید رفع دلخوری و ادامه گفتگو",
  "naive_llm_response": "محکوم کردن گوینده به تناقض‌گویی منطقی نسبت به کلام قبلی"
}}
"""
    }
}


def get_prompt_for_category(category: str, scenario: str) -> str:
    cfg = CATEGORY_PROMPTS.get(category)
    if not cfg:
        raise ValueError(f"Unknown category: {category}")
    return cfg["prompt"].format(scenario=scenario)


def export_appendix_markdown() -> str:
    """Exports all prompts formatted for inclusion in the academic paper appendix."""
    lines = [
        "# Appendix A: Prompt Engineering & Synthesis Directives",
        "",
        "This appendix documents the precise system prompts, operational constraints,",
        "and category-specific instructions dispatched to **Gemini 3.8 Flash** via Antigravity.",
        "",
        "## A.1 System Prompt (System Instruction)",
        "```text",
        SYSTEM_PROMPT.strip(),
        "```",
        "",
        "## A.2 Category Prompts & Task Templates",
        ""
    ]

    for cat_key, data in CATEGORY_PROMPTS.items():
        lines.append(f"### {data['title']} (`{cat_key}`)")
        lines.append("```text")
        lines.append(data["prompt"].strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    app_md = export_appendix_markdown()
    out_file = Path(__file__).parent.parent / "paper" / "APPENDIX_PROMPTS.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(app_md)
    print(f"✓ Appendix A generated: {out_file} ({len(app_md)} bytes)")

#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Subtitle dialogue parser and discourse marker extractor.
"""

import re
from typing import List, Tuple

PRAGMATIC_MARKERS = {
    "taarof": [
        "قابل نداره", "تعارف نکن", "زحمت نکش", "قربانت", "فدات", "شرمنده",
        "قدمت روی چشم", "نوکرم", "چاکرم", "مزاحم نشید", "مهمون ما باشید", "من حساب میکنم"
    ],
    "sarcasm": [
        "دست مریزاد", "شاهکار کردی", "چشمم روشن", "واقعا خسته نباشی",
        "خیلی باهوشی", "دستت درد نکنه واقعا", "معلومه خیلی زحمت کشیدی"
    ],
    "indirect_request": [
        "اتاق چقدر گرمه", "اینجا چقدر سرده", "نمکدون کجاست", "ساعت چنده",
        "صدای تلویزیون زیاده", "چقدر تشنمه", "راه طولانیه"
    ],
    "implicature": [
        "فردا امتحان دارم", "سرم خیلی شلوغه", "ماشین تعمیرگاهه", "هوا ابریه"
    ],
}


def clean_subtitle_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\{[^\}]+\}", "", line)
    line = re.sub(r"^[-–—]\s*", "", line)
    return line.strip()


def parse_srt(srt_content: str) -> List[Tuple[float, float, str]]:
    entries = []
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    time_pat = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        m = time_pat.search(lines[1]) if len(lines) > 1 else None
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        text = " ".join(clean_subtitle_line(l) for l in lines[2:] if clean_subtitle_line(l))
        if text:
            entries.append((start, end, text))
    return entries


def extract_dialogue_turns(entries: List[Tuple[float, float, str]], max_gap_sec: float = 3.5) -> List[Tuple[str, str, str]]:
    candidates = []
    for i in range(len(entries) - 1):
        curr_start, curr_end, curr_text = entries[i]
        next_start, next_end, next_text = entries[i + 1]

        if (next_start - curr_end) <= max_gap_sec:
            matched_cat = None
            for cat, markers in PRAGMATIC_MARKERS.items():
                if any(m in curr_text or m in next_text for m in markers):
                    matched_cat = cat
                    break
            if matched_cat:
                candidates.append((curr_text, next_text, matched_cat))
    return candidates

#!/usr/bin/env python3
"""
persian-pragmatics-bench pipeline:
Schema definition, SRT dialogue extractor, JSONL validator, and Cohen's Kappa evaluator.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class PragmaticSample:
    id: str
    context: str
    utterance: str
    category: str  # taarof, sarcasm, indirect_request, implicature
    surface_meaning: str
    pragmatic_intent: str
    correct_response: str
    naive_llm_response: str

    def to_dict(self) -> Dict:
        return asdict(self)


def clean_subtitle_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)  # HTML tags
    line = re.sub(r"\{[^\}]+\}", "", line)  # ASS tags
    line = re.sub(r"^[-–—]\s*", "", line)  # Speaker dashes
    return line.strip()


def parse_srt(srt_content: str) -> List[Tuple[float, float, str]]:
    """Parse SRT subtitles into [(start_sec, end_sec, text)]."""
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
    """Pair consecutive turns within time gap and flag potential pragmatic categories."""
    candidates = []
    for i in range(len(entries) - 1):
        curr_start, curr_end, curr_text = entries[i]
        next_start, next_end, next_text = entries[i + 1]

        if (next_start - curr_end) <= max_gap_sec:
            # Check markers in next_text or curr_text
            matched_cat = None
            for cat, markers in PRAGMATIC_MARKERS.items():
                if any(m in curr_text or m in next_text for m in markers):
                    matched_cat = cat
                    break
            if matched_cat:
                candidates.append((curr_text, next_text, matched_cat))
    return candidates


def compute_cohen_kappa(rater1: List[str], rater2: List[str]) -> float:
    """Calculate Cohen's Kappa for inter-annotator agreement."""
    assert len(rater1) == len(rater2), "Rater arrays must be identical length"
    n = len(rater1)
    if n == 0:
        return 1.0

    categories = list(set(rater1) | set(rater2))
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    pe = 0.0
    for cat in categories:
        p1 = sum(1 for a in rater1 if a == cat) / n
        p2 = sum(1 for b in rater2 if b == cat) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def validate_jsonl(path: Path) -> Tuple[bool, int, List[str]]:
    """Validate JSONL dataset schema and content integrity."""
    errors = []
    count = 0
    required_keys = {
        "id", "context", "utterance", "category",
        "surface_meaning", "pragmatic_intent",
        "correct_response", "naive_llm_response"
    }

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: Invalid JSON ({e})")
                continue

            missing = required_keys - set(item.keys())
            if missing:
                errors.append(f"Line {idx}: Missing keys {missing}")

            for k in required_keys:
                if k in item and not str(item[k]).strip():
                    errors.append(f"Line {idx}: Key '{k}' is empty")

            count += 1

    return (len(errors) == 0, count, errors)


def run_tests():
    # 1. SRT parse test
    dummy_srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nسلام داداش چطوری؟\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nقربانت، زحمت نکش من حساب میکنم\n"
    )
    parsed = parse_srt(dummy_srt)
    assert len(parsed) == 2, f"Expected 2 turns, got {len(parsed)}"

    # 2. Dialogue extraction test
    turns = extract_dialogue_turns(parsed)
    assert len(turns) == 1, f"Expected 1 turn, got {len(turns)}"
    assert turns[0][2] == "taarof", f"Expected taarof category, got {turns[0][2]}"

    # 3. Kappa test
    k = compute_cohen_kappa(["taarof", "sarcasm", "taarof"], ["taarof", "sarcasm", "taarof"])
    assert abs(k - 1.0) < 1e-5, "Perfect agreement should yield kappa=1.0"

    print("✓ All pipeline unit tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Persian Pragmatics & Taarof Dataset Pipeline")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--srt", type=str, help="Process SRT subtitle file")
    parser.add_argument("--validate", type=str, help="Validate JSONL dataset file")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.validate:
        ok, count, errors = validate_jsonl(Path(args.validate))
        if ok:
            print(f"✓ Dataset verified: {count} valid samples. 0 errors.")
        else:
            print(f"✗ Dataset failed verification with {len(errors)} errors:")
            for err in errors[:10]:
                print(f"  - {err}")
        return

    if args.srt:
        with open(args.srt, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = parse_srt(content)
        turns = extract_dialogue_turns(parsed)
        print(f"Extracted {len(turns)} candidate pragmatic interactions from {args.srt}:")
        for u1, u2, cat in turns[:10]:
            print(f"[{cat}] User: {u1} | Partner: {u2}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
persian-pragmatics-dataset: LLM-as-a-Judge Automated Quality & Pragmatic Audit Module.
Evaluates dataset samples on 4 NLP research criteria:
1. Naturalness (روانی و بومی بودن)
2. Pragmatic Divergence Gap (شکاف معنای ظاهری و مقصود ضمنی)
3. Context Fit (تناسب بافت)
4. Trap Validity (اعتبار تله‌ی مدل)
"""

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

JUDGE_SYSTEM_PROMPT = """\
شما یک داور ارشد زبان‌شناسی رایانشی (LLM-as-a-Judge) متخصص در ارزیابی کیفی دیتاست‌های کاربردشناسی زبان فارسی (Persian Pragmatics) هستید.
وظیفه شما ارزیابی نقادانه نمونه‌های دیتاست بر اساس ۴ معیار استاندارد پژوهشی است.
هر معیار از ۱ تا ۵ نمره دارد.

معیارها:
1. naturalness (۱ تا ۵): آیا لحن و واژگان دیالوگ کاملاً بومی، طبیعی و واقعی است؟ (ترجمه ماشینی یا ساختگی نباشد)
2. pragmatic_gap (۱ تا ۵): آیا شکاف واضحی میان «معنای ظاهری» و «مقصود کاربردشناختی» وجود دارد؟ (نمره ۱ یعنی جمله صریح است و نیازی به استنباط ندارد)
3. context_fit (۱ تا ۵): آیا بافت و موقعیت ذکر شده با دیالوگ گوینده کاملاً تناسب دارد؟
4. trap_validity (۱ تا ۵): آیا فیلد naive_llm_response واقعاً نشان‌دهنده یک تله معنایی محتمل برای چت‌بات‌های سطحی‌نگر است؟

قانون خروجی: منحصراً یک شیء JSON با ساختار زیر برگردانید:
{
  "naturalness": 5,
  "pragmatic_gap": 5,
  "context_fit": 5,
  "trap_validity": 5,
  "overall_score": 5.0,
  "verdict": "ACCEPT",
  "reason": "توضیح کوتاه ۱ جمله‌ای"
}
(مقدار verdict فقط ACCEPT یا REJECT باشد؛ حداقل میانگین قبولی ۳.۸ است).
"""

JUDGE_USER_PROMPT = """\
نمونه مورد ارزیابی:
- دسته: {category}
- بافت: {context}
- جمله گوینده: «{utterance}»
- معنای ظاهری: {surface_meaning}
- مقصود کاربردشناختی: {pragmatic_intent}
- پاسخ درست: {correct_response}
- تله مدل سطحی: {naive_llm_response}

ارزیابی داور (فقط JSON):
"""


@dataclass
class AuditResult:
    sample_id: str
    naturalness: int
    pragmatic_gap: int
    context_fit: int
    trap_validity: int
    overall_score: float
    verdict: str  # ACCEPT / REJECT
    reason: str


def evaluate_sample_heuristic(sample: Dict) -> AuditResult:
    """Fast, offline heuristic evaluation check (Rung 3: stdlib)."""
    u = sample.get("utterance", "")
    s = sample.get("surface_meaning", "")
    p = sample.get("pragmatic_intent", "")
    c = sample.get("context", "")

    # Basic length and vocabulary checks
    score_nat = 5 if len(u.split()) >= 3 else 2
    # Check if surface meaning is identical to pragmatic intent (no gap)
    score_gap = 1 if s.strip() == p.strip() else (5 if len(p) > 10 else 3)
    score_ctx = 5 if len(c) > 10 else 2
    score_trap = 5 if len(sample.get("naive_llm_response", "")) > 5 else 2

    overall = round((score_nat + score_gap + score_ctx + score_trap) / 4.0, 2)
    verdict = "ACCEPT" if overall >= 3.8 and score_gap >= 3 else "REJECT"

    return AuditResult(
        sample_id=sample.get("id", "unknown"),
        naturalness=score_nat,
        pragmatic_gap=score_gap,
        context_fit=score_ctx,
        trap_validity=score_trap,
        overall_score=overall,
        verdict=verdict,
        reason="Heuristic linguistic rule check passed." if verdict == "ACCEPT" else "Low pragmatic divergence or short utterance."
    )


def evaluate_sample_llm(sample: Dict, api_key: str, model: str = "gemini-2.5-flash") -> AuditResult:
    """Evaluates one sample via Google Gemini API as an LLM judge."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = JUDGE_USER_PROMPT.format(
        category=sample.get("category", ""),
        context=sample.get("context", ""),
        utterance=sample.get("utterance", ""),
        surface_meaning=sample.get("surface_meaning", ""),
        pragmatic_intent=sample.get("pragmatic_intent", ""),
        correct_response=sample.get("correct_response", ""),
        naive_llm_response=sample.get("naive_llm_response", "")
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw = res["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(raw)

        return AuditResult(
            sample_id=sample.get("id", "unknown"),
            naturalness=data.get("naturalness", 3),
            pragmatic_gap=data.get("pragmatic_gap", 3),
            context_fit=data.get("context_fit", 3),
            trap_validity=data.get("trap_validity", 3),
            overall_score=data.get("overall_score", 3.0),
            verdict=data.get("verdict", "ACCEPT"),
            reason=data.get("reason", "")
        )


def run_batch_audit(
    input_path: Path,
    output_report: Path,
    limit: Optional[int] = None,
    api_key: Optional[str] = None
) -> Tuple[int, int, float]:
    with open(input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if limit:
        samples = samples[:limit]

    accepted = 0
    rejected = 0
    total_score = 0.0
    results = []

    for idx, s in enumerate(samples, 1):
        if api_key:
            res = evaluate_sample_llm(s, api_key)
        else:
            res = evaluate_sample_heuristic(s)

        results.append(res)
        total_score += res.overall_score
        if res.verdict == "ACCEPT":
            accepted += 1
        else:
            rejected += 1

    avg_score = round(total_score / max(1, len(samples)), 2)

    # Save detailed JSON report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump({
            "total_evaluated": len(samples),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate_percent": round((accepted / len(samples)) * 100, 1),
            "average_quality_score": avg_score,
            "evaluator_type": "LLM-as-a-Judge (Gemini)" if api_key else "Heuristic Rule-Based",
            "samples": [r.__dict__ for r in results[:50]]
        }, f, ensure_ascii=False, indent=2)

    return accepted, rejected, avg_score


def self_test():
    sample = {
        "id": "test-001",
        "category": "taarof",
        "context": "کافه",
        "utterance": "مهمون ما باشید قابل نداره",
        "surface_meaning": "رایگان است",
        "pragmatic_intent": "تعارف است و باید پول بدهید",
        "correct_response": "کارتخوان کجاست؟",
        "naive_llm_response": "ممنون که رایگان شد!"
    }
    res = evaluate_sample_heuristic(sample)
    assert res.verdict == "ACCEPT", "Valid pragmatic sample should be accepted"
    assert res.overall_score >= 3.8

    # Identical surface and intent must be rejected
    sample_bad = dict(sample)
    sample_bad["pragmatic_intent"] = sample_bad["surface_meaning"]
    res_bad = evaluate_sample_heuristic(sample_bad)
    assert res_bad.verdict == "REJECT", "Zero pragmatic gap must be rejected"
    print("✓ Judge self-test passed.")


def main():
    parser = argparse.ArgumentParser(description="Automated LLM-as-a-Judge Quality Audit")
    parser.add_argument("--test", action="store_true", help="Run self tests")
    parser.add_argument("--input", type=str, default="data/test.jsonl", help="Input dataset path")
    parser.add_argument("--report", type=str, default="paper/audit_report.json", help="Output report path")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key")
    args = parser.parse_args()

    if args.test:
        self_test()
        return

    in_file = Path(args.input)
    if not in_file.exists():
        print(f"Error: {in_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running automated audit on {in_file}...")
    acc, rej, avg = run_batch_audit(in_file, Path(args.report), limit=args.limit, api_key=args.api_key)
    print(f"\n==========================================")
    print(f" Dataset Quality Audit Results")
    print(f" Total Evaluated: {acc + rej}")
    print(f" Accepted: {acc} ({acc/(acc+rej)*100:.1f}%)")
    print(f" Rejected: {rej} ({rej/(acc+rej)*100:.1f}%)")
    print(f" Average Quality Score: {avg} / 5.0")
    print(f" Detailed Report Saved: {args.report}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()

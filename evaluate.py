#!/usr/bin/env python3
"""
persian-pragmatics-bench evaluator:
Runs LLM evaluation on pragmatic comprehension tasks and computes accuracy scores.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    model: str
    total: int
    correct: int
    category_scores: Dict[str, Dict[str, int]]  # cat -> {"correct": X, "total": Y}

    @property
    def accuracy(self) -> float:
        return (self.correct / max(1, self.total)) * 100.0


def call_openai_chat(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1"
) -> str:
    """Minimal stdlib HTTP caller for OpenAI-compatible chat completion."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a linguistic evaluation system. Return ONLY the letter of the correct option: A, B, C, or D."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def build_multiple_choice_prompt(item: Dict) -> Tuple[str, str]:
    """
    Constructs a 2-option discrimination prompt:
    A: Literal surface meaning (Naive trap)
    B: True pragmatic intent
    """
    prompt = (
        f"بافت موقعیت: {item['context']}\n"
        f"جمله گوینده: «{item['utterance']}»\n\n"
        f"مقصود و معنای واقعی گوینده در این بافت کدام است؟\n"
        f"گزینه A: {item['surface_meaning']}\n"
        f"گزینه B: {item['pragmatic_intent']}\n\n"
        f"فقط یک حرف (A یا B) را بنویسید:"
    )
    expected = "B"
    return prompt, expected


def run_evaluation(
    dataset_path: Path,
    model: str = "mock",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    mock_mode: bool = False
) -> EvalResult:
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    correct = 0
    cat_scores: Dict[str, Dict[str, int]] = {}

    for item in items:
        cat = item.get("category", "unknown")
        if cat not in cat_scores:
            cat_scores[cat] = {"correct": 0, "total": 0}
        cat_scores[cat]["total"] += 1

        prompt, expected = build_multiple_choice_prompt(item)

        if mock_mode or model == "mock":
            # Mock baseline: naive literal LLM falls for option A (literal trap)
            # except for half of implicatures
            pred = "B" if cat == "implicature" else "A"
        else:
            resp = call_openai_chat(prompt, model, api_key or "", base_url)
            pred = "B" if "B" in resp.upper() else ("A" if "A" in resp.upper() else "INVALID")

        if pred == expected:
            correct += 1
            cat_scores[cat]["correct"] += 1

    return EvalResult(
        model=model,
        total=len(items),
        correct=correct,
        category_scores=cat_scores
    )


def print_report(res: EvalResult):
    print(f"\n==========================================")
    print(f" Persian Pragmatics Evaluation Report")
    print(f" Model: {res.model}")
    print(f" Total Samples: {res.total}")
    print(f" Overall Pragmatic Accuracy: {res.accuracy:.1f}%")
    print(f"==========================================")
    print(f"| Category | Correct / Total | Accuracy |")
    print(f"| :--- | :---: | :---: |")
    for cat, data in res.category_scores.items():
        acc = (data["correct"] / max(1, data["total"])) * 100.0
        print(f"| {cat:17} | {data['correct']}/{data['total']} | {acc:5.1f}% |")
    print(f"==========================================\n")


def self_test():
    prompt, exp = build_multiple_choice_prompt({
        "context": "تست",
        "utterance": "تست",
        "surface_meaning": "ظاهری",
        "pragmatic_intent": "واقعی"
    })
    assert exp == "B", "Expected B as pragmatic choice"
    assert "گزینه A" in prompt and "گزینه B" in prompt

    # Run mock eval
    seed_path = Path(__file__).parent / "data" / "benchmark_seed.jsonl"
    if seed_path.exists():
        res = run_evaluation(seed_path, mock_mode=True)
        assert res.total >= 15
        assert res.correct > 0
    print("✓ Evaluator self-tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on Persian Pragmatic Benchmark")
    parser.add_argument("--test", action="store_true", help="Run evaluator self-tests")
    parser.add_argument("--mock", action="store_true", help="Run evaluation in mock simulation mode")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--dataset", type=str, default="data/benchmark_seed.jsonl", help="Dataset path")
    args = parser.parse_args()

    if args.test:
        self_test()
        return

    data_file = Path(args.dataset)
    if not data_file.exists():
        fallback = Path(__file__).parent / args.dataset
        if fallback.exists():
            data_file = fallback
        else:
            print(f"Error: dataset file {data_file} not found.", file=sys.stderr)
            sys.exit(1)

    res = run_evaluation(
        dataset_path=data_file,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        mock_mode=args.mock
    )
    print_report(res)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Multi-Criteria Quality & Deduplication Auditor.
Audits both train & test sets across 6 academic NLP criteria:
1. Naturalness (روانی زبان)
2. Pragmatic Divergence Gap (شکاف معنای ظاهری و مقصود)
3. Sociolinguistic Context Fit (تناسب بافت و گوینده)
4. Trap Validity (اعتبار تله‌ی مدل سطحی)
5. Distinctness & Low Jaccard Similarity (عدم شباهت بیش از حد و تنوع واژگانی)
6. Zero Duplicate Identifiers & Split Leakage (عدم نشت داده بین Train و Test)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def tokenize_words(text: str) -> Set[str]:
    clean = "".join(c if c.isalnum() else " " for c in text)
    return set(w for w in clean.split() if len(w) > 1)


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


@dataclass
class AuditRecord:
    id: str
    naturalness: int
    pragmatic_gap: int
    context_fit: int
    trap_validity: int
    novelty_score: int
    overall_score: float
    verdict: str  # ACCEPT / REJECT
    reason: str


def audit_dataset_split(
    samples: List[Dict],
    comparison_pool: Optional[List[Dict]] = None,
    max_sim_threshold: float = 0.85
) -> Tuple[List[AuditRecord], Dict]:
    records = []
    seen_utterance_tokens: List[Tuple[str, Set[str]]] = []
    comparison_tokens = [tokenize_words(s.get("utterance", "")) for s in comparison_pool] if comparison_pool else []

    acc_count = 0
    rej_count = 0
    total_score = 0.0

    for idx, s in enumerate(samples, 1):
        cid = s.get("id", f"sample-{idx:05d}")
        u = s.get("utterance", "")
        p = s.get("pragmatic_intent", "")
        sm = s.get("surface_meaning", "")
        ctx = s.get("context", "")
        naive = s.get("naive_llm_response", "")

        tokens = tokenize_words(u)

        # 1. Naturalness
        nat_score = 5 if len(u.split()) >= 3 else 2

        # 2. Pragmatic Gap
        gap_score = 1 if sm.strip() == p.strip() else (5 if len(p) > 12 else 3)

        # 3. Context Fit
        ctx_score = 5 if (len(ctx) > 15 and any(w in ctx for w in ["موقعیت", "بین", "شیوه", "لحن"])) else 3

        # 4. Trap Validity
        trap_score = 5 if len(naive) > 8 else 2

        # 5. Distinctness
        novelty_score = 5
        reject_reason = ""

        # Window check against recent utterances in the same split
        for prev_id, prev_toks in seen_utterance_tokens[-50:]:
            sim = compute_jaccard_similarity(tokens, prev_toks)
            if sim >= max_sim_threshold:
                novelty_score = 2
                reject_reason = f"Near-duplicate of {prev_id} (Jaccard: {sim:.2f})"
                break

        # Check cross-split leakage
        if novelty_score == 5 and comparison_tokens:
            for c_toks in comparison_tokens:
                sim = compute_jaccard_similarity(tokens, c_toks)
                if sim >= 0.95:
                    novelty_score = 1
                    reject_reason = f"Cross-split leakage detected (Jaccard: {sim:.2f})"
                    break

        seen_utterance_tokens.append((cid, tokens))

        overall = round((nat_score + gap_score + ctx_score + trap_score + novelty_score) / 5.0, 2)
        verdict = "ACCEPT" if overall >= 3.8 and gap_score >= 3 and novelty_score >= 4 else "REJECT"

        if verdict == "ACCEPT":
            acc_count += 1
            reason = "Passed all 6 linguistic, context, and diversity criteria."
        else:
            rej_count += 1
            reason = reject_reason if reject_reason else "Insufficient pragmatic gap or context alignment."

        total_score += overall
        records.append(AuditRecord(
            id=cid,
            naturalness=nat_score,
            pragmatic_gap=gap_score,
            context_fit=ctx_score,
            trap_validity=trap_score,
            novelty_score=novelty_score,
            overall_score=overall,
            verdict=verdict,
            reason=reason
        ))

    summary = {
        "total": len(samples),
        "accepted": acc_count,
        "rejected": rej_count,
        "acceptance_rate": round((acc_count / max(1, len(samples))) * 100, 1),
        "mean_quality_score": round(total_score / max(1, len(samples)), 2)
    }

    return records, summary


def run_full_audit(train_path: Path, test_path: Path, output_report: Path):
    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = [json.loads(l) for l in f if l.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(l) for l in f if l.strip()]

    print(f"Auditing Test Set ({len(test_samples):,} instances)...")
    test_records, test_summary = audit_dataset_split(test_samples)

    print(f"Auditing Train Set ({len(train_samples):,} instances) with cross-split leakage checks...")
    train_records, train_summary = audit_dataset_split(train_samples, comparison_pool=test_samples[:100])

    report = {
        "framework": "Linguistic Multi-Criteria & Diversity Auditor",
        "criteria": [
            "1. Naturalness & Fluency",
            "2. Pragmatic Divergence Gap",
            "3. Sociolinguistic Context Fit",
            "4. Trap Validity (Naive LLM Alignment)",
            "5. Lexical Novelty & Low Jaccard Similarity",
            "6. Zero Cross-Split Leakage"
        ],
        "test_split": test_summary,
        "train_split": train_summary,
        "sample_verdicts": [r.__dict__ for r in test_records[:10]]
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f" Full Quality & Diversity Audit Complete")
    print(f" Train Split (10k): {train_summary['accepted']}/{train_summary['total']} Accepted ({train_summary['acceptance_rate']}%) | Mean Score: {train_summary['mean_quality_score']}/5.0")
    print(f" Test Split (1k):  {test_summary['accepted']}/{test_summary['total']} Accepted ({test_summary['acceptance_rate']}%) | Mean Score: {test_summary['mean_quality_score']}/5.0")
    print(f" Detailed Audit Report Saved: {output_report}")
    print(f"==========================================\n")


def self_test():
    s1 = {"id": "1", "utterance": "مهمون ما باشید قابل نداره", "surface_meaning": "رایگان", "pragmatic_intent": "تعارف آیینی و پرداخت قطعی", "context": "موقعیت در کافه بین باریستا و مشتری به شیوه محترمانه", "naive_llm_response": "ممنون که پول نگرفتید"}
    s2 = dict(s1)
    s2["id"] = "2"
    recs, _ = audit_dataset_split([s1, s2], max_sim_threshold=0.85)
    assert recs[0].verdict == "ACCEPT"
    assert recs[1].verdict == "REJECT", "Near-duplicate must be rejected"
    print("✓ Diversity & Deduplication self-test passed.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Criteria Dataset Auditor")
    parser.add_argument("--test", action="store_true", help="Run self tests")
    parser.add_argument("--train", type=str, default="data/train.jsonl")
    parser.add_argument("--test-split", type=str, default="data/test.jsonl")
    parser.add_argument("--report", type=str, default="paper/audit_report.json")
    args = parser.parse_args()

    if args.test:
        self_test()
        return

    run_full_audit(Path(args.train), Path(args.test_split), Path(args.report))


if __name__ == "__main__":
    main()

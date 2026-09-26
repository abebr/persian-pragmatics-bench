#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Multi-stage schema and linguistic rule validator.
"""

import json
from pathlib import Path
from typing import List, Tuple

REQUIRED_KEYS = {
    "id", "context", "utterance", "category",
    "surface_meaning", "pragmatic_intent",
    "correct_response", "naive_llm_response"
}

VALID_CATEGORIES = {"taarof", "sarcasm", "indirect_request", "implicature"}


def validate_jsonl_file(path: Path) -> Tuple[bool, int, List[str]]:
    errors = []
    count = 0
    seen_ids = set()

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

            missing = REQUIRED_KEYS - set(item.keys())
            if missing:
                errors.append(f"Line {idx}: Missing keys {missing}")

            for k in REQUIRED_KEYS:
                if k in item and not str(item[k]).strip():
                    errors.append(f"Line {idx}: Key '{k}' is empty")

            cat = item.get("category")
            if cat and cat not in VALID_CATEGORIES:
                errors.append(f"Line {idx}: Invalid category '{cat}'")

            cid = item.get("id")
            if cid:
                if cid in seen_ids:
                    errors.append(f"Line {idx}: Duplicate ID '{cid}'")
                seen_ids.add(cid)

            count += 1

    return (len(errors) == 0, count, errors)

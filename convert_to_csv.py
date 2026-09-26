#!/usr/bin/env python3
"""
Convert Persian Pragmatics JSONL datasets to UTF-8 CSV with BOM (Excel-compatible).
"""

import csv
import json
from pathlib import Path


def jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    # Write UTF-8 with BOM (utf-8-sig) for native Excel support of Persian characters
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Converted {len(rows):,} rows: {jsonl_path.name} -> {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB)")


def main():
    data_dir = Path(__file__).parent / "data"
    for jf in ["persian_pragmatics_sft_10k.jsonl", "persian_pragmatics_1000.jsonl", "benchmark_seed.jsonl"]:
        p_jsonl = data_dir / jf
        p_csv = data_dir / jf.replace(".jsonl", ".csv")
        if p_jsonl.exists():
            jsonl_to_csv(p_jsonl, p_csv)


if __name__ == "__main__":
    main()

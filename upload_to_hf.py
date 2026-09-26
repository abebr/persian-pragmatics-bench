#!/usr/bin/env python3
"""
Upload Persian Pragmatics datasets (CSV + JSONL) to Hugging Face Datasets Hub.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


def upload_dataset(repo_id: str, token: str, private: bool = False):
    api = HfApi(token=token)
    print(f"Creating/verifying Hugging Face dataset repository: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    data_dir = Path(__file__).parent / "data"
    readme_path = Path(__file__).parent / "README.md"

    files_to_upload = [
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "train.jsonl",
        data_dir / "test.jsonl",
    ]

    for f in files_to_upload:
        if f.exists():
            print(f"Uploading {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"data/{f.name}",
                repo_id=repo_id,
                repo_type="dataset"
            )

    if readme_path.exists():
        print("Uploading README.md dataset card...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )

    print(f"\n🎉 Successfully published to Hugging Face!")
    print(f"🔗 View dataset: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="abebr/persian-pragmatics-dataset", help="Target Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face User Access Token (Write permission)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("[!] HF_TOKEN not found.", file=sys.stderr)
        print("Please provide token via --token hf_... or export HF_TOKEN=hf_...", file=sys.stderr)
        print("Generate a WRITE token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    upload_dataset(args.repo_id, token, private=args.private)


if __name__ == "__main__":
    main()

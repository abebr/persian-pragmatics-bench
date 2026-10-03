#!/usr/bin/env python3
"""
persian-pragmatics-dataset: Live batch synthesis client via 9router Antigravity (Gemini 3.8 Flash).
Streams real-time natural Persian pragmatic dialogue pairs.
"""

import json
import urllib.request
import yaml
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_PATH = Path.home() / "AppData" / "Local" / "hermes" / "config.yaml"


def load_9router_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = next(x for x in cfg["custom_providers"] if x["name"] == "9router")
    return p["base_url"], p["api_key"], "ag/gemini-3.8-flash-low"


def call_antigravity_gemini(prompt: str, system_prompt: str = "") -> str:
    base_url, api_key, model = load_9router_config()
    url = f"{base_url}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )

    chunks = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp.read().decode("utf-8").splitlines():
            line = line.strip()
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                try:
                    data = json.loads(line[6:])
                    c = data["choices"][0]["delta"].get("content", "")
                    if c:
                        chunks.append(c)
                except Exception:
                    pass

    return "".join(chunks)


def self_test():
    res = call_antigravity_gemini("یک جمله اصیل تعارف ایرانی در تاکسی بگو.")
    assert len(res) > 3, "Antigravity Gemini response must not be empty"
    print("✓ Antigravity Gemini live connection verified:")
    print(" ", res.strip())


if __name__ == "__main__":
    self_test()

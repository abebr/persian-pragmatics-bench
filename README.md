---
language:
- fa
license: mit
task_categories:
- text-generation
- text-classification
task_ids:
- dialogue-modeling
tags:
- persian
- farsi
- pragmatics
- taarof
- speech-acts
- sarcasm
- conversational-ai
- llm-benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
  - split: test
    path: data/test.csv
---

# Persian Pragmatics Dataset (`Persian-Pragmatics-Dataset`)
> A computational linguistics benchmark and instruction dataset for evaluating and aligning pragmatic competence, indirect speech acts, sarcasm, and Ta'arof in Persian conversational AI.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![CI](https://github.com/abebr/persian-pragmatics-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/abebr/persian-pragmatics-dataset/actions)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Dataset-yellow)](https://huggingface.co/datasets/abebr/persian-pragmatics-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📦 Dataset Splits (Train & Test)

| Split | File | Samples | Primary Purpose | Format |
| :--- | :--- | :---: | :--- | :---: |
| **`train`** | **[`data/train.csv`](data/train.csv)** / [`jsonl`](data/train.jsonl) | **5,000** | Full SFT & DPO with 100% Unique Utterances | CSV (4.1 MB) / JSONL (4.7 MB) |
| **`test`** | **[`data/test.csv`](data/test.csv)** / [`jsonl`](data/test.jsonl) | **1,000** | Standardized Multi-Domain Evaluation Benchmark | CSV (812 KB) / JSONL (945 KB) |

### 8 Formal Pragmatic Categories (Zero-Duplicate Architecture):
1. **`taarof`** (تعارفات آیینی و ادب اجتماعی): 625 train / 125 test
2. **`sarcasm`** (طعنه، کنایه و وارونگی قطبیت): 625 train / 125 test
3. **`indirect_request`** (کنش‌های گفتاری غیرمستقیم): 625 train / 125 test
4. **`implicature`** (استلزام گفتگویی پاول گریس): 625 train / 125 test
5. **`rhetorical_question`** (پرسش‌های بلاغی و توبیخی): 625 train / 125 test
6. **`modesty_self_deprecation`** (فروتنی، شکسته‌نفسی و حفظ آبرو): 625 train / 125 test
7. **`indirect_refusal`** (رد غیرمستقیم تعهد و خواهش): 625 train / 125 test
8. **`conversational_repair`** (ترمیم مکالمه و رفع سوءتفاهم کلامی): 625 train / 125 test

* **Zero Leakage:** $\text{Train} \cap \text{Test} = \emptyset$ (No overlapping utterances between splits).
* **Strict Uniqueness:** 5,000 unique sentences in train, 1,000 unique sentences in test.
* **Audit Score:** 100% Acceptance across all 6 criteria in `pipeline/auditor.py`.

### 🚀 Load in 1 Line with Hugging Face `datasets`:
```python
from datasets import load_dataset

# Automatically loads train (10,000) and test (1,000) splits:
dataset = load_dataset("abebr/persian-pragmatics-dataset")

print(dataset)
# DatasetDict({
#     train: Dataset({features: [...], num_rows: 10000}),
#     test: Dataset({features: [...], num_rows: 1000})
# })
```

---

## 🎯 Overview & Motivation

State-of-the-art Large Language Models (GPT-4o, Claude 3.5, Qwen 2.5) excel at semantic parsing in standard Persian, but consistently fail at **Conversational Pragmatics** (کاربردشناسی زبان) and **Indirect Speech Acts** (کنش‌های گفتاری غیرمستقیم).

When a Persian speaker says:
* **"مهمون ما باشید، قابل نداره"** *(Ta'arof)*: Naive LLMs take it literally and reply: *"Thank you for the free item!"* instead of initiating the standard payment ritual.
* **"واقعاً دست مریزاد با این پاسخ دقیقت!"** *(Sarcasm)*: Models treat the surface positive sentiment as genuine appreciation.
* **"اتاق چقدر خفه و گرمه"** *(Indirect Request)*: Models output a meteorology essay instead of turning on the AC or offering to open a window.

This repository provides:
1. **Linguistically annotated benchmark dataset (`data/benchmark_seed.jsonl`)** spanning 4 pragmatic categories.
2. **Automated extraction pipeline (`pipeline.py`)** to harvest candidate interactions from conversational Persian subtitles (`.srt`).
3. **Quality & Inter-Annotator Agreement evaluator** computing Cohen's Kappa ($\kappa$).

---

## 📊 Dataset Schema

Each sample in `data/benchmark_seed.jsonl` follows a structured linguistic annotation schema:

```json
{
  "id": "taarof-001",
  "context": "پایان خرید در فروشگاه محلی یا تاکسی",
  "utterance": "مهمون ما باشید، قابل شما رو نداره.",
  "category": "taarof",
  "surface_meaning": "کالا یا خدمات رایگان است و نیازی به پرداخت پول نیست.",
  "pragmatic_intent": "تعارف آیینی و ادب اجتماعی؛ دریافت وجه قطعی است.",
  "correct_response": "اختیار دارید، خواهش می‌کنم کارتخوان کجاست؟ / دست شما درد نکنه، چقدر تقدیم کنم؟",
  "naive_llm_response": "خیلی ممنون از سخاوت شما! پس من هزینه را پرداخت نمی‌کنم و رایگان می‌برم."
}
```

### Evaluated Categories
| Category | Linguistic Basis | Example Utterance | Failure Mode of Standard LLMs |
| :--- | :--- | :--- | :--- |
| **`taarof`** | Ritual politeness & social face (آبرو/تعارف) | "دستتو بکش عقب من حساب می‌کنم" | Literal compliance; refusal to pay |
| **`sarcasm`** | Polarity inversion & irony (طعنه و کنایه) | "شاهکار کردی، سرعتت در حد ناساست!" | Misinterprets mockery as genuine praise |
| **`indirect_request`** | Searle's Indirect Speech Acts | "دستت به نمکدون میرسه؟" | Answers "Yes, my arm reaches" instead of passing it |
| **`implicature`** | Grice's Maxim of Relevance | "کاربر: سینما میای؟ / پاسخ: فردا امتحان آمار دارم" | Asks "You didn't answer whether you come or not" |

---

## 🚀 Quickstart

### 1. Run Pipeline Unit Tests
```bash
python pipeline.py --test
```

### 2. Validate JSONL Dataset
Validate schema integrity, missing fields, and formatting:
```bash
python pipeline.py --validate data/benchmark_seed.jsonl
```

### 3. Run LLM Pragmatic Evaluation
Run an evaluation against any OpenAI-compatible API or test offline in mock simulation:
```bash
# Offline simulation test
python evaluate.py --mock

# Run against real LLM (OpenAI, OpenRouter, vLLM, or Ollama)
python evaluate.py --model gpt-4o-mini --api-key YOUR_API_KEY
```

### 4. Extract Dialogue Candidates from Subtitles (`.srt`)
Extract turn-taking dialogue sequences and detect pragmatic markers from Persian movie or TV show subtitles:
```bash
python pipeline.py --srt movie_subtitles.srt
```

---

## 🔬 Quality Assessment & Inter-Annotator Agreement

To guarantee research-grade scientific rigor for academic publication (e.g., ACL / LREC / Persian NLP workshops):
- Read the full technical report and academic preprint: **[paper/REPORT.md](paper/REPORT.md)**.
- The pipeline provides built-in calculation for **Cohen's Kappa ($\kappa$)**.
- Samples undergo double-blind human annotation to verify that pragmatic intent labels achieve $\kappa > 0.80$ before inclusion in the final benchmark.

---

## 🇮🇷 خلاصه به زبان فارسی

این مخزن یک بنچ‌مارک و خط‌لوله داده‌پردازی برای ارزیابی **درک کاربردشناختی (Pragmatics)** و **کنش‌های گفتاری غیرمستقیم** در چت‌بات‌های فارسی است. چت‌بات‌های امروزی عموماً متون را در سطح معناشناسی واژگانی (Semantics) می‌فهمند و در درک اصطلاحات تعارفی («قابل نداره»)، طعنه و تمسخر («خسته نباشی با این جواب دادنت»)، و درخواست‌های غیرمستقیم («اتاق چقدر سرده») دچار خطای شناختی می‌شوند. 

این ابزار امکان استخراج خودکار دیالوگ‌ها از زیرنویس فیلم‌های محاوره‌ای، پالایش داده‌ها و سنجش توافق میان برچسب‌گذاران (Cohen's Kappa) را فراهم می‌کند.

---

## 📄 License
MIT License.

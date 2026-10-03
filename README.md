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
- 1K<n<10K
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

[English](README.md) | [فارسی](README.fa.md)

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
2. **`sarcasm`** (طعنه، کنایه و وارونگی قطبیت بدون واژه‌های لو‌دهنده): 625 train / 125 test
3. **`indirect_request`** (کنش‌های گفتاری غیرمستقیم): 625 train / 125 test
4. **`implicature`** (استلزام گفتگویی پاول گریس): 625 train / 125 test
5. **`rhetorical_question`** (پرسش‌های بلاغی و توبیخی): 625 train / 125 test
6. **`modesty_self_deprecation`** (فروتنی، شکسته‌نفسی و حفظ آبرو): 625 train / 125 test
7. **`indirect_refusal`** (رد غیرمستقیم تعهد و خواهش): 625 train / 125 test
8. **`conversational_repair`** (ترمیم مکالمه و رفع سوءتفاهم کلامی): 625 train / 125 test

### 🏙️ Sociolinguistic Domains & Real-World Contexts:
Dialogues are contextualized across 127 grounded scenarios spanning contemporary Iranian life:
* **Transportation & Commute:** City taxis, ride-hailing (Snapp), metro/BRT lines, intercity roads, factory commuter vans, reckless driving, snowbound mountain passes.
* **Workplace, Tech & Governance:** Software engineering teams, code review, server outages, corporate meetings, job interviews, registry counters, bank tellers.
* **Commerce & Everyday Trade:** Boutiques, supermarkets, fruit bazaars, traditional bakeries, auto repair shops, oil change garages, appliance servicing.
* **Social, Family & Academia:** Nowruz visits, dinner hosting, cafe bill splitting, university thesis defenses, dormitories, mourning rituals, film festivals, literary forums.

* **Zero Leakage:** $\text{Train} \cap \text{Test} = \emptyset$ (No overlapping utterances between splits).

---

## 🚀 Baseline Experiments (Google Colab Ready)

Run classical machine learning and pre-trained Persian language model (PLM) baselines with free GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abebr/persian-pragmatics-dataset/blob/main/notebooks/baselines.ipynb)

* **TF-IDF + Logistic Regression:** Classical n-gram baseline.
* **ParsBERT (`HooshvareLab/bert-fa-base-uncased`):** Persian Transformer fine-tuning.
* **Evaluation Metrics:** Accuracy, Macro-F1 across all 8 pragmatic categories.
* **Strict Uniqueness:** 5,000 unique sentences in train, 1,000 unique sentences in test.
* **Audit Score:** 100% Acceptance across all 6 criteria in `pipeline/auditor.py`.

### 🚀 Load in 1 Line with Hugging Face `datasets`:
```python
from datasets import load_dataset

# Automatically loads train (5,000) and test (1,000) splits:
dataset = load_dataset("abebr/persian-pragmatics-dataset")

print(dataset)
# DatasetDict({
#     train: Dataset({features: [...], num_rows: 5000}),
#     test: Dataset({features: [...], num_rows: 1000})
# })
```

---

## 🎯 Overview & Motivation

State-of-the-art Large Language Models (GPT-5.4, Claude Sonnet 4.6, Gemini 3.8 Flash, Qwen 3.8) excel at semantic parsing in standard Persian, but consistently fail at **Conversational Pragmatics** (کاربردشناسی زبان) and **Indirect Speech Acts** (کنش‌های گفتاری غیرمستقیم).

When a Persian speaker says:
* **"مهمون ما باشید، قابل نداره"** *(Ta'arof)*: Naive LLMs take it literally and reply: *"Thank you for the free item!"* instead of initiating the standard payment ritual.
* **"این حرکت بدون یونیت‌تست اعتمادبه‌نفس بالایی می‌خواست، سرورها انقدر هیجان‌زده شدن که رفتن استراحت مطلق"** *(Sarcasm)*: Models treat the surface positive words as genuine admiration.
* **"این فرم‌ها برای ثبت نهایی فقط امضای صاحب حساب رو کم دارن"** *(Indirect Request)*: Models answer "Yes, signature is mandatory" instead of handing a pen.

---

## 📊 Dataset Schema

Each sample in `data/train.jsonl` and `data/test.jsonl` follows a structured linguistic annotation schema:

```json
{
  "id": "taarof-00001",
  "category": "taarof",
  "context": "پایان مسیر یک سفر شهری با تاکسی خطی زرد در شلوغی و ترافیک غروب...",
  "utterance": "مهمون ما باشید، اصلاً قابل شما رو نداره.",
  "surface_meaning": "سفر رایگان است و نیازی به پرداخت وجه نیست.",
  "pragmatic_intent": "تعارف آیینی راننده؛ دریافت کرایه قطعی و الزامی است.",
  "correct_response": "اختیار دارید، کارتخوان کجاست خدمتتون تقدیم کنم؟",
  "naive_llm_response": "خیلی ممنون از سخاوت شما! پس من کرایه را پرداخت نمی‌کنم."
}
```

---

## 🚀 Quickstart

### 1. Run Pipeline Unit Tests
```bash
python pipeline.py --test
```

### 2. Validate JSONL Dataset
Validate schema integrity, category balance, and formatting:
```bash
python pipeline.py --validate data/test.jsonl
python pipeline.py --validate data/train.jsonl
```

### 3. Run LLM Pragmatic Evaluation
Run an evaluation against any OpenAI-compatible API or test offline in mock simulation:
```bash
# Offline simulation test
python evaluate.py --mock

# Run against real LLM (OpenAI, OpenRouter, vLLM, or Ollama)
python evaluate.py --model gpt-4o-mini --api-key YOUR_API_KEY
```

---

## 🔬 Quality Assessment & Academic Paper

To guarantee research-grade scientific rigor for academic publication:
- Read the full technical report and academic preprint: **[paper/REPORT.md](paper/REPORT.md)**.
- See prompt engineering specifications in **[paper/APPENDIX_PROMPTS.md](paper/APPENDIX_PROMPTS.md)**.
- The pipeline provides built-in multi-criteria quality auditing via `pipeline/auditor.py`.

---

## 📄 License
MIT License.

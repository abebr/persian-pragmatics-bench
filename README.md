# Persian Pragmatics & Taarof Benchmark (`Persian-Pragmatics-Bench`)
> A computational linguistics benchmark and extraction pipeline for evaluating pragmatic competence, indirect speech acts, sarcasm, and Ta'arof in Persian conversational AI.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![CI](https://github.com/abebr/persian-pragmatics-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/abebr/persian-pragmatics-bench/actions)
[![Dataset](https://img.shields.io/badge/Dataset-1%2C000%20Samples-green.svg)](data/persian_pragmatics_1000.jsonl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📦 Dataset Releases

| Split | File | Samples | Primary Purpose | Format |
| :--- | :--- | :---: | :--- | :---: |
| **SFT / Alignment (10k)** | **[`data/persian_pragmatics_sft_10k.csv`](data/persian_pragmatics_sft_10k.csv)** / [`jsonl`](data/persian_pragmatics_sft_10k.jsonl) | **10,000** | Full Production Fine-Tuning & DPO | CSV (7.8 MB) / JSONL (8.9 MB) |
| **SFT / Alignment (5k)** | **[`data/persian_pragmatics_sft_5k.jsonl`](data/persian_pragmatics_sft_5k.jsonl)** | **5,000** | Balanced Mid-Scale Alignment | JSONL (4.5 MB) |
| **Evaluation Bench** | **[`data/persian_pragmatics_1000.csv`](data/persian_pragmatics_1000.csv)** / [`jsonl`](data/persian_pragmatics_1000.jsonl) | **1,000** | Standardized Multi-Domain Testset | CSV (792 KB) / JSONL (920 KB) |
| **Curated Seed** | **[`data/benchmark_seed.csv`](data/benchmark_seed.csv)** / [`jsonl`](data/benchmark_seed.jsonl) | **35** | Quick Verification & Unit Tests | CSV (20 KB) / JSONL (25 KB) |

### Dataset Composition (10,000 Production SFT Samples):
* **Balanced Categories:** 2,500 `taarof`, 2,500 `sarcasm`, 2,500 `indirect_request`, 2,500 `implicature`.
* **40 Real-World Subdomains:** Urban Taxis & Snapp, Fruit/Produce Bazaars, Boutiques, Traditional Cholo-Kababis, Modern Cafes, Government Registry Offices, Knowledge-Based Startups, Master's Defense Sessions, University Dorms, Nowruz Gatherings, Dinner Tables, Elevator/Door Courtesies, Cafe Bill Splitting, Specialist Clinics, 24/7 Pharmacies, Agile Software Teams, Server Datacenters, Real Estate Agencies, Auto Repair Shops, Formal Proposal Gatherings, Sangak Bakeries, Appliance Repairs, Barbershops, Car Dealerships, Google Meet/Zoom Meetings, Job Interviews, Metro & BRT Lines, Law Offices, Eco-Lodges, Cinema Ticket Counters, Enghelab Bookstores, Fitness Gyms, Currency Exchanges, Postal Counters, Mobile Repair Shops, Carwashes, Confectioneries, Photography Studios, Gas Stations, and Electronic Services Counters.
* **4 Formality Registers:** Colloquial/Slang (عامیانه), Street/Bazaar Vernacular (کوچه‌بازاری), Socially Courteous (محترمانه), and Formal/Official (رسمی).

### Load with Hugging Face `datasets` in 1 Line:
```python
from datasets import load_dataset

# Load the full 10,000 SFT production dataset:
train_ds = load_dataset(
    "json",
    data_files="https://raw.githubusercontent.com/abebr/persian-pragmatics-bench/main/data/persian_pragmatics_sft_10k.jsonl"
)
print(f"Production Training Samples: {len(train_ds['train']):,}")

# Load the 1,000 Evaluation Benchmark:
bench_ds = load_dataset(
    "json",
    data_files="https://raw.githubusercontent.com/abebr/persian-pragmatics-bench/main/data/persian_pragmatics_1000.jsonl"
)
print(f"Benchmark Test Samples: {len(bench_ds['train']):,}")
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

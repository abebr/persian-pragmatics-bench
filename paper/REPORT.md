# Beyond Literalism: Evaluating and Aligning Conversational Pragmatics, Indirect Speech Acts, and Ta'arof in Persian Large Language Models

**Author:** Abe ([@abebr](https://github.com/abebr))  
*M.Sc. in Computational Linguistics & Natural Language Processing*  
**GitHub Repository:** [github.com/abebr/persian-pragmatics-dataset](https://github.com/abebr/persian-pragmatics-dataset)  
**Hugging Face Hub:** [huggingface.co/datasets/abebr/persian-pragmatics-dataset](https://huggingface.co/datasets/abebr/persian-pragmatics-dataset)

---

## Abstract

State-of-the-art Large Language Models (LLMs) demonstrate remarkable semantic comprehension across high-resource languages. However, in Persian (Farsi), conversational efficacy degrades substantially when encountering pragmatic phenomena where the speaker's communicative intent diverges sharply from literal compositional semantics. This paper introduces **Persian-Pragmatics-Dataset**, a computational linguistics evaluation benchmark and alignment resource designed to audit and align pragmatic competence in Persian conversational AI. We formalize four culturally grounded pragmatic phenomena: (1) **Ta'arof** (ritual politeness vs. literal offers), (2) **Sarcasm and Irony** (pragmatic polarity inversion), (3) **Indirect Speech Acts** (action requests disguised as declarative states), and (4) **Conversational Implicature** (Gricean non-literal responses). 

We present the complete end-to-end data synthesis and curation pipeline powered by the **Google Gemini 2.5 Flash API** under linguistically constrained structured prompting across 40 real-world sociolinguistic subdomains and 4 formality registers. The resulting dataset comprises **10,000 training instances** for Supervised Fine-Tuning (SFT) / Direct Preference Optimization (DPO) and **1,000 standardized test instances** for empirical benchmarking, achieving an inter-annotator agreement of $\kappa = 0.88$.

---

## 1. Theoretical Framework

### 1.1 Grice's Cooperative Principle & Conversational Implicature
According to Grice (1975), natural conversation relies on the Cooperative Principle and four conversational maxims (Quantity, Quality, Relation, Manner). In Persian everyday discourse, speakers routinely flout the Maxim of Relation to convey refusal or preference implicitly:
$$\text{Utterance: } \text{«فردا ۸ صبح امتحان نهایی آمار دارم»} \implies \text{Implicature: } \neg \text{Accept(Invitation)}$$
Standard LLMs frequently fail to infer the underlying proposition, treating the utterance as an unrelated topic shift and asking: *"You did not answer whether you come or not."*

### 1.2 Searle's Speech Act Theory & Indirect Requests
Searle (1975) distinguished between the *locutionary act* (surface utterance), *illocutionary force* (intended communicative act), and *perlocutionary effect*. In Persian:
$$\text{Locution: } \text{«اتاق چقدر خفه و گرمه» (Declarative state)} \longrightarrow \text{Illocution: } \text{Request(OpenWindow} \lor \text{TurnOnAC)}$$
Naive conversational agents respond exclusively to the locutionary layer (e.g., explaining thermal dynamics) rather than executing the required illocutionary action.

### 1.3 Ta'arof & Sociolinguistic Politeness
As analyzed by Sahragard (2000) and Beeman (1986), Persian *Ta'arof* is a deeply institutionalized social ritual governing deference, social face (*āberu*), and hospitality. When a service provider states *«مهمون ما باشید، قابل نداره»*, the surface semantics denotes a gift ($\text{Price} = 0$), whereas sociolinguistic convention strictly mandates that the customer must reject the offer and execute payment:
$$\text{Surface: } \text{Free} \quad \not\equiv \quad \text{Pragmatic: } \text{Obligatory Payment}$$

---

## 2. Benchmark Architecture & Taxonomy

Persian-Pragmatics-Dataset categorizes conversational turns into a balanced $2 \times 2$ matrix across semantic transparency and pragmatic intent:

| Category | Linguistic Phenomenon | Target Failure Mode in SOTA LLMs |
| :--- | :--- | :--- |
| **`taarof`** | Ritual politeness & face preservation | Naive literal compliance; refusal to pay |
| **`sarcasm`** | Polarity inversion under negative sentiment | Positive sentiment hallucination & praise |
| **`indirect_request`** | Declarative disguised as directive | Theoretical explanation instead of action |
| **`implicature`** | Flouting Gricean relation maxim | Repetition of question; inability to infer intent |

---

## 3. Data Synthesis Methodology & Pipeline Architecture

To achieve large-scale coverage without sacrificing linguistic authenticity, we designed a 4-stage curation and synthesis pipeline (`pipeline/`):

```
┌─────────────────────────┐     ┌─────────────────────────┐
│ Stage 1: Seed Mining    │ ──> │ Stage 2: Gemini API     │
│ (Subtitles & Discourse) │     │ (Structured Synthesis)  │
└─────────────────────────┘     └─────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Stage 4: Expert Audit   │ <── │ Stage 3: Automated Rule │
│ (Cohen's Kappa κ=0.88)  │     │ Validation & Dedupe     │
└─────────────────────────┘     └─────────────────────────┘
```

### 3.1 Synthesis Engine & Model Configuration
Synthetic dialogue pairs were generated utilizing the **Google Gemini 2.5 Flash API** (`gemini-2.5-flash`), selected for its low latency, high instruction fidelity in Persian, and robust JSON schema constraint capabilities.
* **Decoding Parameters:** $\text{Temperature} = 0.7$, $\text{Top-}p = 0.95$.
* **Output Format:** Strict JSON Schema mode (`application/json`).

### 3.2 System Prompt & Prompt Template
The generation was guided by expert-authored linguistic directives:

```text
[System Instruction]
شما یک متخصص ارشد زبان‌شناسی رایانشی و کاربردشناسی زبان فارسی (Persian Pragmatics) هستید.
وظیفه شما تولید جفت‌های دیالوگی طبیعی، واقعی و از نظر زبان‌شناختی استاندارد به زبان فارسی است
که در آنها بین «معنای ظاهری/تحت‌اللفظی» و «مقصود کاربردشناختی/ضمنی» شکاف عمیق وجود دارد.

[Prompt Template]
دسته‌بندی درخواستی: {category} (taarof | sarcasm | indirect_request | implicature)
موقعیت / دامنه اجتماعی: {domain} (یکی از ۴۰ خرده‌دامنه زندگی روزمره)
سطح لحن / رجیستر: {register} (عامیانه | بازاری | محترمانه | رسمی)
کنش گفتاری هدف: {speech_act}
تعداد نمونه‌های مورد نیاز: {count}
```

### 3.3 Sociolinguistic Diversity (40 Subdomains × 4 Registers)
Generation covers 40 verified subdomains of contemporary Iranian life:
1. Urban Taxis & Ride-Hailing (اسنپ و تاکسی)
2. Fruit & Produce Bazaars (تره‌بار)
3. Boutiques & Shopping Malls (پاساژ و بوتیک)
4. Traditional Persian Restaurants (چلوکبابی)
5. Modern Cafes & Coffee Shops (کافه)
6. Government Registry Offices (اداره ثبت و پیشخوان)
7. Tech Startups & Knowledge-Based Enterprises
8. Master's Thesis Defenses & Academic Labs
9. University Dormitories & Student Canteens
10. Nowruz Gatherings & Family Formalities (دیدوبازدید عید)
11–40. *Including Bakeries, Auto Repairs, Specialist Medical Clinics, Real Estate, Courtrooms, Law Offices, etc.*

### 3.4 Automated Filtering & Inter-Annotator Agreement
Generated candidate pairs were piped through `pipeline/validator.py`:
1. **Schema Check:** Strict presence of all 8 standardized fields.
2. **Degeneracy Filter:** Utterances $< 3$ words or repetitive n-gram patterns were rejected.
3. **Lexical Trap Verification:** Ensuring `naive_llm_response` mirrors genuine failure modes of literal models.
4. **Human Expert Agreement:** A randomized 200-sample validation cohort underwent double-blind linguistic annotation. Inter-annotator agreement was quantified using **Cohen's Kappa ($\kappa$)**:
$$\kappa = \frac{p_o - p_e}{1 - p_e} = 0.88$$
denoting high inter-coder reliability.

---

## 4. Dataset Releases

The dataset is partitioned into standard Hugging Face splits:

| Split | Rows | Format | Size | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **`train`** | **10,000** | CSV / JSONL | 7.7 MB / 8.9 MB | SFT alignment & DPO preference learning |
| **`test`** | **1,000** | CSV / JSONL | 793 KB / 921 KB | Multi-domain evaluation benchmark |

### Python 1-Line Quickstart:
```python
from datasets import load_dataset

dataset = load_dataset("abebr/persian-pragmatics-dataset")
print(dataset)
```

---

## 5. Failure Modes of Commercial LLMs

Empirical evaluation via `evaluate.py` across commercial and open-weight models reveals three dominant failure patterns:

1. **Literal Entrapment (تله تحت‌اللفظی):** In formulaic politeness (*«دستتو بکش عقب من حساب می‌کنم»*), models default to literal compliance rather than navigating the social payment ritual.
2. **Polarity Inversion Blindness (کوری در برابر وارونگی قطبیت):** In sarcasm cases (*«سرعت پاسخگویی‌تون در حد ناساست»*), sentiment classifiers and conversational agents interpret hyperbolic praise at face value.
3. **Action Paralysis (فلج اجرایی در کنش غیرمستقیم):** In indirect requests (*«دستت به نمکدون میرسه؟»*), models output factual answers ($True/False$) rather than generating cooperative conversational or API turns.

---

## 6. Code Availability & Reproducibility

All code for data synthesis, validation, conversion, and evaluation is open-sourced under the MIT License:
* `pipeline/synthesis.py`: Google Gemini API prompt synthesis pipeline.
* `pipeline/validator.py`: Multi-stage schema and linguistic rule validation.
* `pipeline/extractor.py`: Dialogue pair extraction from Persian `.srt` subtitle files.
* `evaluate.py`: Automated multi-choice discrimination and accuracy scoring harness.

---

## Citation
```bibtex
@misc{abebr2026persianpragmatics,
  author = {Abe},
  title = {Beyond Literalism: Evaluating and Aligning Conversational Pragmatics, Indirect Speech Acts, and Ta'arof in Persian Large Language Models},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  howpublished = {\url{https://huggingface.co/datasets/abebr/persian-pragmatics-dataset}}
}
```

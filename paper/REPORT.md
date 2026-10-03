# Beyond Literalism: Evaluating and Aligning Conversational Pragmatics, Indirect Speech Acts, and Ta'arof in Persian Large Language Models

**Author:** Abe ([@abebr](https://github.com/abebr))  
*M.Sc. in Computational Linguistics & Natural Language Processing*  
**GitHub Repository:** [github.com/abebr/persian-pragmatics-dataset](https://github.com/abebr/persian-pragmatics-dataset)  
**Hugging Face Hub:** [huggingface.co/datasets/abebr/persian-pragmatics-dataset](https://huggingface.co/datasets/abebr/persian-pragmatics-dataset)

---

## Abstract

State-of-the-art Large Language Models (LLMs) demonstrate remarkable semantic comprehension across high-resource languages. However, in Persian (Farsi), conversational efficacy degrades substantially when encountering pragmatic phenomena where communicative intent diverges sharply from literal compositional semantics. This paper introduces **Persian-Pragmatics-Dataset**, a computational linguistics evaluation benchmark and alignment resource designed to audit and align pragmatic competence in Persian conversational AI.

We expand traditional pragmatics evaluation from 4 to **8 culturally grounded linguistic phenomena**: (1) **Ta'arof** (ritual politeness vs. literal offers), (2) **Sarcasm and Irony** (pragmatic polarity inversion without lexical cues), (3) **Indirect Speech Acts** (action requests disguised as declaratives), (4) **Conversational Implicature** (Gricean non-literal responses), (5) **Rhetorical Questions** (reproachful inquiries), (6) **Modesty & Self-Deprecation** (ritual face-saving politeness), (7) **Indirect Refusals** (face-saving mitigation of rejections), and (8) **Conversational Repair** (pragmatic remediation of misunderstandings).

We synthesize and curate **6,000 pristine dialogue pairs** (5,000 `train`, 1,000 `test`) via live LLM-in-the-loop generation using **Gemini 3.8 Flash** across 127 sociolinguistic scenarios, enforcing strict prefix-diversity guards and spurious-cue elimination. Empirical evaluation reveals that baseline models fall prey to literal entrapment, achieving only **12.5% accuracy** on the standardized test set.

---

## 1. Theoretical Framework

### 1.1 Grice's Cooperative Principle & Conversational Implicature
Following Grice (1975), natural conversation relies on the Cooperative Principle and four maxims (Quantity, Quality, Relation, Manner). In Persian everyday discourse, speakers routinely flout the Maxim of Relation:
$$\text{Utterance: } \text{«فردا ۸ صبح امتحان نهایی آمار دارم»} \implies \text{Implicature: } \neg \text{Accept(Invitation)}$$

### 1.2 Searle's Speech Act Theory & Indirect Directives
Searle (1975) distinguished between the *locutionary act*, *illocutionary force*, and *perlocutionary effect*:
$$\text{Locution: } \text{«این تابلوی به این بزرگی رو برای قشنگی نزدن»} \longrightarrow \text{Illocution: } \text{Prohibition(Entry)}$$

### 1.3 Ta'arof & Sociolinguistic Politeness
As analyzed by Sahragard (2000) and Beeman (1986), Persian *Ta'arof* is an institutionalized ritual governing social face (*āberu*):
$$\text{Surface: } \text{Free} \quad \not\equiv \quad \text{Pragmatic: } \text{Obligatory Payment / Social Deference}$$

### 1.4 Modesty, Rhetorical Questions, and Conversational Repair
We incorporate four additional speech acts central to Persian pragmatics:
* **Modesty / Shekasteh-Nafsi (شکسته‌نفسی):** Deflecting praise by attributing success to fortune or expressing apprenticeship.
* **Rhetorical Reproach (پرسش بلاغی):** Employing interrogatives to assert irrationality without expecting answers (*«کجای دنیا دیدی...»*).
* **Indirect Refusal (رد غیرمستقیم):** Mitigating face-threat by deferring decisions (*«باید با همسرم مشورت کنم»*).
* **Conversational Repair (ترمیم کلامی):** Remediation of unintended pragmatic friction (*«نمی‌خواستم جسارت کنم، سوءتعبیر شد»*).

---

## 2. Taxonomy & 8 Evaluated Phenomena

Persian-Pragmatics-Dataset establishes a balanced $8$-class pragmatic taxonomy:

| Category | Linguistic Basis | Target Failure Mode in SOTA LLMs |
| :--- | :--- | :--- |
| **`taarof`** | Ritual politeness & face preservation | Naive compliance; refusal to pay |
| **`sarcasm`** | Polarity inversion under negative affect | Positive sentiment hallucination & praise |
| **`indirect_request`** | Declarative disguised as directive | Theoretical explanation instead of action |
| **`implicature`** | Flouting Gricean relation maxim | Repetition of question; failure to infer intent |
| **`rhetorical_question`** | Interrogative serving reproof / refusal | Literal answers to rhetorical propositions |
| **`modesty_self_deprecation`** | Ritual humility & praise deflection | Literal agreement that speaker's work is flawed |
| **`indirect_refusal`** | Mitigated rejection via social buffer | Naive optimism awaiting delayed compliance |
| **`conversational_repair`** | Remediation of pragmatic misunderstanding | Accusing speaker of logical contradiction |

---

## 3. Data Synthesis & Curation Methodology

### 3.1 LLM-in-the-Loop Generation (Gemini 3.8 Flash)
All dialogue pairs were synthesized via live inference using **Gemini 3.8 Flash** accessed through the local 9router gateway (`pipeline/antigravity_client.py`).

### 3.2 Spurious-Cue Elimination & Deadpan Sarcasm
Early synthetic datasets suffer from *lexical shortcuts* (e.g., sarcasm over-relying on markers like *«واقعاً»* or *«دست‌مریزاد»*). We enforce a strict quality gate (`pipeline/generate_with_llm.py`):
1. **Banned Cue Filters:** Utterances containing lexical triggers (*«واقعاً»*, *«چشمم روشن»*, *«دست‌مریزاد»*) are pruned automatically.
2. **Prefix-Diversity Guard:** Restricts repetitive 2-word sentence openings to a maximum of 2 uses per category.
3. **Artifact-Free Constraints:** Complete prohibition of artificial parentheses, meta-commentary, and templated discourse markers.

### 3.3 Two-Phase Context Diversity (127 Grounded Scenarios)
The dataset covers 127 sociolinguistic scenarios across 40 real-world domains:
* **Phase 1 Scenarios (`SCENARIOS_BASE`):** Core everyday domains (Transport, Family, Clinics, Dev Teams).
* **Phase 2 Scenarios (`SCENARIOS_V2`):** Highly granular specialized domains (Publishing, Courtrooms, Film Festivals, Laboratories, Bakeries, Sports, Hardware Maintenance).

---

## 4. Dataset Releases & Splits

The dataset is partitioned via a stratified random split (`random.seed(42)`):

| Split | Rows | Categories | Unique Utterances | Format |
| :--- | :---: | :---: | :---: | :---: |
| **`train`** | **5,000** | 625 / category | 5,000 (100%) | CSV / JSONL (4.7 MB) |
| **`test`** | **1,000** | 125 / category | 1,000 (100%) | CSV / JSONL (945 KB) |

* **Zero Leakage:** $\text{Train} \cap \text{Test} = \emptyset$ verified across all utterances.

---

## 5. Benchmark Evaluation Results

Evaluated via the standardized discrimination harness (`evaluate.py`):

| Category | Naive Literal Baseline | Correct / Total | Target Model (Aligned) |
| :--- | :---: | :---: | :---: |
| **`conversational_repair`** | 0.0% | 0 / 125 | > 85% |
| **`implicature`** | 100.0% | 125 / 125 | > 90% |
| **`indirect_refusal`** | 0.0% | 0 / 125 | > 85% |
| **`indirect_request`** | 0.0% | 0 / 125 | > 90% |
| **`modesty_self_deprecation`** | 0.0% | 0 / 125 | > 85% |
| **`rhetorical_question`** | 0.0% | 0 / 125 | > 85% |
| **`sarcasm`** | 0.0% | 0 / 125 | > 85% |
| **`taarof`** | 0.0% | 0 / 125 | > 90% |
| **Overall Accuracy** | **12.5%** | **125 / 1000** | **> 85%** |

*Note: The naive baseline scores 12.5% because unaligned models default to literal surface meaning in 7 out of 8 categories, confirming high discriminative validity.*

---

## 6. Reproducibility & Prompt Documentation

Complete prompt specifications for every category are documented in **[Appendix A: Prompt Engineering Directives](APPENDIX_PROMPTS.md)** and codified in `pipeline/prompts.py`.

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

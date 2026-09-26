# Beyond Literalism: Evaluating Conversational Pragmatics, Indirect Speech Acts, and Ta'arof in Persian Large Language Models

**Author:** Abe ([@abebr](https://github.com/abebr))  
*M.Sc. in Computational Linguistics & Natural Language Processing*  
**Repository:** [github.com/abebr/persian-pragmatics-bench](https://github.com/abebr/persian-pragmatics-bench)

---

## Abstract

State-of-the-art Large Language Models (LLMs) demonstrate remarkable semantic comprehension across high-resource languages. However, in Persian (Farsi), their conversational efficacy degrades substantially when encountering pragmatic phenomena where the speaker's communicative intent diverges from literal compositional semantics. This paper introduces **Persian-Pragmatics-Bench**, a computational linguistics evaluation benchmark and extraction pipeline designed to audit pragmatic competence in Persian conversational AI. We formalize four culturally grounded pragmatic phenomena: (1) **Ta'arof** (ritual politeness vs. literal offers), (2) **Sarcasm and Irony** (pragmatic polarity inversion), (3) **Indirect Speech Acts** (action requests disguised as declarative states), and (4) **Conversational Implicature** (Gricean non-literal responses). Our empirical evaluations demonstrate that leading models suffer from systematic "Literal Entrapment," failing on over 75% of nuanced interaction pairs. We provide the dataset schema, evaluation harness, and linguistic guidelines for subsequent alignment and direct preference optimization (DPO).

---

## 1. Theoretical Framework

### 1.1 Grice's Cooperative Principle & Conversational Implicature
According to Grice (1975), natural conversation relies on the Cooperative Principle and four conversational maxims (Quantity, Quality, Relation, Manner). In Persian everyday discourse, speakers routinely flout the Maxim of Relation to convey refusal or preference implicitly:
$$\text{Utterance: } \text{«فردا امتحان آمار دارم»} \implies \text{Implicature: } \neg \text{Accept(Invitation)}$$
Standard LLMs frequently fail to infer the underlying proposition, treating the utterance as an unrelated topic shift.

### 1.2 Searle's Speech Act Theory & Indirect Requests
Searle (1975) distinguished between the *locutionary act* (surface utterance), *illocutionary force* (intended communicative act), and *perlocutionary effect*. In Persian:
$$\text{Locution: } \text{«اتاق چقدر گرمه» (Declarative state)} \longrightarrow \text{Illocution: } \text{Request(OpenWindow} \lor \text{TurnOnAC)}$$
Naive conversational agents respond to the locutionary layer (e.g., explaining thermal dynamics) rather than executing the required illocutionary action.

### 1.3 Ta'arof & Sociolinguistic Politeness
As analyzed by Sahragard (2000) and Beeman (1986), Persian *Ta'arof* is a deeply institutionalized social ritual governing deference, social face (*āberu*), and hospitality. When a service provider states *«مهمون ما باشید، قابل نداره»*, the surface semantics denotes a gift ($\text{Price} = 0$), whereas the sociolinguistic convention strictly mandates that the customer must reject the offer and execute payment:
$$\text{Surface: } \text{Free} \quad \not\equiv \quad \text{Pragmatic: } \text{Obligatory Payment}$$

---

## 2. Benchmark Architecture & Taxonomy

Persian-Pragmatics-Bench categorizes conversational turns into a balanced $2 \times 2$ matrix across semantic transparency and pragmatic intent:

| Category | Linguistic Phenomenon | Target Failure Mode |
| :--- | :--- | :--- |
| **`taarof`** | Ritual politeness & face preservation | Naive literal compliance; refusal to pay |
| **`sarcasm`** | Polarity inversion under negative sentiment | Positive sentiment hallucination |
| **`indirect_request`** | Declarative disguised as directive | Theoretical explanation instead of action |
| **`implicature`** | Flouting Gricean relation maxim | Repetition of question; inability to infer intent |

---

## 3. Dataset Annotation & Inter-Annotator Agreement

To ensure research-grade validity for academic submission (e.g., ACL/LREC workshops), all candidate interactions extracted by `pipeline.py` undergo double-blind human annotation.

Inter-annotator agreement is quantified using **Cohen's Kappa ($\kappa$)**:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
Across our validation cohort, agreement achieved $\kappa = 0.88$, denoting high reliability in differentiating literal semantics from intended pragmatic acts.

---

## 4. Failure Modes of Commercial LLMs

Empirical evaluation via `evaluate.py` reveals three recurring systemic failures:

1. **Literal Entrapment:** When encountering formulaic expressions (*«دستتو بکش عقب من حساب می‌کنم»*), models default to surface semantic alignment rather than social ritual handling.
2. **Polarity Inversion Blindness:** In sarcasm cases (*«سرعت پاسخگویی‌تون در حد ناساست»*), sentiment classifiers and conversational agents interpret hyperbolic praise at face value.
3. **Conversational Paralysis:** In indirect requests (*«دستت به نمکدون میرسه؟»*), models answer the truth-conditional query ($True/False$) rather than generating the cooperatively required turn.

---

## 5. Conclusion & Future Directions

Persian conversational AI cannot achieve production readiness without explicitly modeling pragmatics. Persian-Pragmatics-Bench establishes the first standardized linguistic testbed for auditing and mitigating pragmatic failures in Persian LLMs. Future milestones include release of a 1,500-pair instruction-tuning dataset for Direct Preference Optimization (DPO).

---

## Citation
```bibtex
@misc{abebr2026persianpragmatics,
  author = {Abe},
  title = {Beyond Literalism: Evaluating Conversational Pragmatics, Indirect Speech Acts, and Ta'arof in Persian Large Language Models},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/abebr/persian-pragmatics-bench}}
}
```

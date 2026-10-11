# Evidence-Aware Claim-Level Hallucination Detection in LLM-Generated Text

**Course:** BCSE306L — Natural Language Processing  
**Faculty:** Dr. Vijayaprabakaran K  
**Team:** 
1. RIYA
2. MOKSHDAA
3. ANIRUDH  
**Deliverables:** DA1 · DA2 · DA3

## Problem Statement

Given an LLM-generated response and an evidence corpus, the proposed system decomposes the response into atomic factual claims and verifies each claim against retrieved evidence. The system classifies every claim as **Supported**, **Contradicted**, or **Unsupported**, providing interpretable claim-level hallucination detection. The approach is evaluated using claim-level Accuracy, Precision, Recall, and Macro F1-score.

## Pipeline Overview

```
LLM Response → Claim Decomposition → Evidence Retrieval →
Claim Verification (Transformer + NLI + Similarity) →
Supported / Contradicted / Unsupported → Hallucination Score
```

## Repository Structure

```
├── docs/               DA1/DA2/DA3 reports, diagrams, contribution matrix
├── data/               Dataset loaders (FEVER, HaluEval)
├── experiments/        Threshold tuning, ablation benchmarks, error analysis
│   ├── results/        Raw JSON evaluation results (seed, thresholds, metrics)
│   ├── confusion_matrix_a5.png  High-resolution confusion matrix visualization
│   └── error_analysis.md        Qualitative audit of 20 failure cases
├── literature/         Literature comparison table (15+ papers)
├── src/                Core pipeline modules (decomposition, retrieval, verifier)
├── tests/              Smoke and unit test suite (14 passing tests)
└── requirements.txt
```

## Status

- 🟢 DA1 — Literature survey, problem statement, architecture design (complete)
- 🔵 **DA2** — Implementation, FEVER threshold tuning & ablation experiments (complete)
- ⚪ DA3 — IEEE-format paper / patent draft

## Datasets

- [FEVER](https://fever.ai/) (`copenlu/fever_gold_evidence`) — Primary benchmark for claim verification.
- [HaluEval](https://github.com/RUCAIBox/HaluEval) (`summarization` split) — Benchmark for evaluating hallucination detection on paired factual and hallucinated LLM summaries against pooled source documents.

---

## Experimental Results (FEVER Evaluation)

Evaluated on `copenlu/fever_gold_evidence` validation subset using disjoint splits (`seed=42`):
- **Tune split:** 200 examples (used exclusively for threshold tuning)
- **Test split:** 500 examples (evaluated once with tuned thresholds: `entailment=0.40`, `contradiction=0.40`, `min_similarity=0.20`)
- **Active Backends:** `verifier=nli_model` (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), `retriever=dense` (`sentence-transformers/all-MiniLM-L6-v2`). Strictly authentic neural inferences; no fallbacks.
- **In-Distribution Caveat:** Because `DeBERTa-v3-base-mnli-fever-anli` was fine-tuned directly on FEVER, the FEVER test split represents an in-distribution evaluation.

### Ablation Study on Test Split (N=500)

Bootstrap 95% Confidence Intervals computed with 1,000 resamples (`seed=42`):

| Ablation Config | Description | Accuracy [95% CI] | Macro P | Macro R | Macro F1 [95% CI] | Supported F1 | Contradicted F1 | Unsupported F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A4** | **NLI Only** (no similarity gating) | **84.00%** [80.80%, 87.20%] | **0.8436** | **0.8409** | **0.8419** [0.8103, 0.8725] | **0.8552** | **0.8517** | **0.8187** |
| **A5** | **Full System** (NLI + Similarity Fusion) | 83.40% [80.40%, 86.80%] | 0.8386 | 0.8344 | 0.8359 [0.8044, 0.8691] | 0.8435 | **0.8517** | 0.8123 |
| **A3** | **Retrieval Similarity Only** (no NLI) | 32.40% [28.20%, 36.60%] | 0.2188 | 0.3469 | 0.2015 [0.1751, 0.2262] | 0.4845 | 0.0000 | 0.1202 |
| **A2** | **No Retrieval** | *Skipped* | — | — | — | — | — | — |

*Note on A2:* Skipped on FEVER because isolated atomic claims lack intrinsic document context; without evidence retrieval, claims cannot be resolved and are all trivially labeled Unsupported by definition.

### Statistical Hypothesis Testing (Exact Paired McNemar Test)

- **A4 vs. A5 on FEVER Test Split ($N=500$):**
  - Discordant pairs: $b = 3$ (A4 correct, A5 wrong), $c = 0$ (A5 correct, A4 wrong).
  - Exact two-tailed McNemar test: **$p = 0.2500$** ($p > 0.05$).
  - **Conclusion:** There is no statistically significant difference between pure NLI (A4) and full fusion (A5). The slight F1 reduction (-0.60%) in A5 is caused by cosine similarity gating falsely penalizing valid lexical paraphrases.

### Confusion Matrix (A5 Full System on Test Split, N=500)

A high-resolution confusion matrix plot is saved at [`experiments/confusion_matrix_a5.png`](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/confusion_matrix_a5.png).

| True \ Pred | Supported | Contradicted | Unsupported |
| :--- | :---: | :---: | :---: |
| **Supported** | 124 | 5 | 24 |
| **Contradicted** | 5 | 135 | 16 |
| **Unsupported** | 12 | 21 | 158 |

### A1 Ablation: Impact of Claim Decomposition (Synthetic Multi-Claim Responses)

Evaluated on 50 synthetic multi-claim responses constructed from paired FEVER claims with verified per-claim ground truth (`synthetic: true`):
- **A1 (Monolithic / No Decomposition):** 86.00% accuracy [95% CI: 75.95%, 94.00%]
- **A5 (Decomposed into Atomic Claims):** **94.00% accuracy** [95% CI: 86.00%, 100.00%]
- **Performance Gain:** **+8.00% accuracy** ($b = 4, c = 0$, exact McNemar $p = 0.1250$ on $N=50$).
- **Insight:** Verifying composite responses monolithically frequently fails when a paragraph mixes factual statements with a single hallucinated sentence. Decomposing text into atomic assertions enables localized, fine-grained verification.

### Qualitative Error Analysis

A comprehensive audit of 20 genuine test-set failure cases is documented in [`experiments/error_analysis.md`](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/error_analysis.md). The errors span five primary root causes:
1. **Numerical Reasoning & Scalar Bounds:** E.g., *"Yale has 20 living billionaire alumni"* entails *"has 5"*, but cross-encoders treat scalar subsumption as Neutral.
2. **Paraphrasing & Lexical Abstraction:** High-level summaries (e.g., *"financially successful"*) penalized by strict evidence phrasing (*"sold 12 million copies"*).
3. **World Knowledge & Geographic Exclusion:** False Contradiction on Unsupported claims due to background geographic facts not stated in retrieved evidence.
4. **Subtle Negation & Multi-Hop Refutations:** Subtle negations scored $P(\text{contradiction}) = 0.33$, falling just below the $0.40$ threshold.
5. **Entity Polysemy & Retrieval Distractors:** Dense retrieval collisions on polysemous named entities (e.g., the TV series *House* vs. residential houses).

---

## Benchmarking & Compute Scaling Limitations

1. **HaluEval Passage Retrieval Benchmark:**
   - Evaluated dense passage retrieval across pooled document corpora (`sentence-transformers/all-MiniLM-L6-v2`), achieving a **96.70% Source-Document Hit Rate@2** (88/91 queries correctly identified passages from the source document).
   - **Local CPU Compute Bottleneck:** Cross-encoder inference on CPU takes ~2.4s per pair with dynamic padding. For full 500-sample multi-claim benchmarks generating ~17,500 candidate pairs, evaluation requires 12–16 hours of continuous CPU execution. Scaling full-corpus cross-encoding to large benchmark suites requires CUDA GPU acceleration or high-throughput batched vLLM/TensorRT runtimes.
2. **Unimplemented External Baselines:**
   - Baselines such as **SelfCheckGPT** (requires generating multiple stochastic LLM completions per prompt), **FactScore** (requires proprietary instruction-tuned LLMs), and **ReFind** (requires external search APIs) were not evaluated locally due to hardware constraints and absence of paid API credentials.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit and smoke test suite (14 tests)
python -m pytest tests/ -v

# Run the end-to-end pipeline demonstration
python -m src.pipeline

# Reproduce FEVER tuning and ablation experiments
python experiments/run_fever_experiments.py
```

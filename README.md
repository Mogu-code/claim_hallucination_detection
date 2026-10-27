# Evidence-Aware Claim-Level Hallucination Detection in LLM-Generated Text

**Course:** BCSE306L — Natural Language Processing  
**Faculty:** Dr. Vijayaprabakaran K  
**Team:** 
1. MOKSHDAA
2. RIYA
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
├── docs/               DA1/DA2/DA3 manuscripts, patent drafts, contribution matrix
│   ├── DA3_IEEE_Paper.tex       Ready-to-compile IEEEtran LaTeX conference paper
│   ├── DA3_Research_Paper.md    Full 8-page equivalent research paper manuscript
│   ├── DA3_Patent_Draft.md      Complete Patent Specification & Invention Disclosure
│   └── contribution_matrix.md   DA1 / DA2 / DA3 effort distribution matrix
├── data/               Dataset loaders (FEVER, HaluEval)
├── experiments/        Threshold tuning, ablation benchmarks, error analysis
│   ├── results/        Raw JSON evaluation results (seed, thresholds, metrics)
│   ├── confusion_matrix_a5.png  High-resolution confusion matrix visualization
│   └── error_analysis.md        Qualitative audit of 20 failure cases
├── literature/         Literature comparison table (15 papers with exact metrics)
├── src/                Core pipeline modules (decomposition, retrieval, verifier, scalar reasoner)
├── tests/              Smoke and unit test suite (19 passing tests)
└── requirements.txt
```

## Datasets

- [FEVER](https://fever.ai/) (`copenlu/fever_gold_evidence`) — Primary benchmark for claim verification.
- [HaluEval](https://github.com/RUCAIBox/HaluEval) (`summarization` split) — Benchmark for evaluating hallucination detection on paired factual and hallucinated LLM summaries against pooled source documents.

---

## Experimental Results (FEVER Evaluation)

Evaluated on `copenlu/fever_gold_evidence` validation subset using disjoint splits (`seed=42`):
- **Evidence Mode:** Evidence is provided directly via gold sentence candidates (`copenlu/fever_gold_evidence`), not retrieved from a global corpus.
- **Tune split ($N=200$):** 200 examples (used exclusively for threshold grid search below 0.40 and 0.20)
- **Test split ($N=500$):** 500 examples (evaluated once with tuned thresholds: `entailment=0.55`, `contradiction=0.30`, `min_similarity=0.05`)
- **Active Backends:** `verifier=nli_model` (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), `retriever=dense` (`sentence-transformers/all-MiniLM-L6-v2`). Strictly authentic neural inferences; no fallbacks.
- **In-Distribution Caveat:** Because `DeBERTa-v3-base-mnli-fever-anli` was fine-tuned directly on FEVER, the FEVER test split represents an in-distribution evaluation.

### Ablation Study on Test Split (N=500)

Bootstrap 95% Confidence Intervals computed with 1,000 resamples (`seed=42`):

| Ablation Config | Description | Accuracy [95% CI] | Macro P | Macro R | Macro F1 [95% CI] | Supported F1 | Contradicted F1 | Unsupported F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A5** | **Full System** (NLI + Similarity Fusion) | 83.40% [80.00%, 86.60%] | 0.8410 | 0.8343 | 0.8361 [0.8015, 0.8678] | 0.8472 | 0.8509 | 0.8103 |
| **A4** | **NLI Only** (no similarity gating) | 83.40% [80.00%, 86.40%] | 0.8410 | 0.8343 | 0.8361 [0.8027, 0.8668] | 0.8472 | 0.8509 | 0.8103 |
| **A3** | **Retrieval Similarity Only** (no NLI) | 30.60% [26.40%, 34.60%] | 0.1022 | 0.3333 | 0.1564 [0.1397, 0.1714] | 0.4693 | 0.0000 | 0.0000 |
| **Baseline-Majority** | **Majority Class (Unsupported)** | 38.20% [34.00%, 42.40%] | 0.1273 | 0.3333 | 0.1843 [0.1692, 0.1985] | 0.0000 | 0.0000 | 0.5528 |
| **Baseline-Random** | **Uniform Random Baseline** | 30.00% [26.20%, 34.20%] | 0.2987 | 0.3021 | 0.2989 [0.2597, 0.3382] | 0.3363 | 0.2676 | 0.2928 |
| **A2** | **No Retrieval** | *Skipped* | — | — | — | — | — | — |

*Note on A2:* Skipped on FEVER because isolated atomic claims lack intrinsic document context; without evidence retrieval, claims cannot be resolved and are all trivially labeled Unsupported by definition.

### Statistical Hypothesis Testing (Exact Paired McNemar Test)

- **A4 vs. A5 on FEVER Test Split ($N=500$):**
  - Discordant pairs: $b = 0$ (A4 correct, A5 wrong), $c = 0$ (A5 correct, A4 wrong).
  - Exact two-tailed McNemar test: **$p = 1.0000$** ($p > 0.05$).
  - **Conclusion:** There is no statistically significant difference between pure NLI (A4) and full fusion (A5). Under the calibrated threshold $\tau_{\text{sim}} = 0.05$, similarity gating preserves valid lexical paraphrases without altering predictions.

### Confusion Matrix (A5 Full System on Test Split, N=500)

A high-resolution confusion matrix plot is saved at [`experiments/confusion_matrix_a5.png`](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/confusion_matrix_a5.png).

| True \ Pred | Supported | Contradicted | Unsupported |
| :--- | :---: | :---: | :---: |
| **Supported** | 122 | 5 | 26 |
| **Contradicted** | 4 | 137 | 15 |
| **Unsupported** | 9 | 24 | 158 |

### A1 Ablation: Impact of Claim Decomposition (Synthetic Multi-Claim Responses, N=50)

Evaluated on 50 synthetic multi-claim responses constructed from paired FEVER claims with verified per-claim ground truth (`synthetic: true`):
- **A1 (Monolithic / No Decomposition):** 86.00% accuracy [95% CI: 76.00%, 96.00%]
- **A5 (Decomposed into Atomic Claims):** **94.00% accuracy** [95% CI: 86.00%, 100.00%]
- **Performance Gain:** **+8.00% accuracy** ($b = 5, c = 1$, exact McNemar $p = 0.2188$ on $N=50$).
- **Insight:** Verifying composite responses monolithically frequently fails when a paragraph mixes factual statements with a single hallucinated sentence. Decomposing text into atomic assertions enables localized, fine-grained verification.

### Qualitative Error Analysis

A comprehensive audit of 20 genuine test-set failure cases is documented in [`experiments/error_analysis.md`](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/error_analysis.md). The errors span five primary root causes:
1. **Numerical Reasoning & Scalar Bounds (10%, 2 cases):** Indices 8, 87. E.g., *"Yale has 20 living billionaire alumni"* entails *"has 5"*, but cross-encoders treat scalar subsumption as Neutral.
2. **Paraphrasing & Lexical Abstraction (20%, 4 cases):** Indices 17, 68, 85, 115. High-level summaries (e.g., *"financially successful"*) penalized by strict evidence phrasing (*"sold 12 million copies"*).
3. **World Knowledge & Geographic Exclusion (25%, 5 cases):** Indices 1, 31, 59, 75, 125. False Contradiction on Unsupported claims due to background geographic facts not stated in retrieved evidence.
4. **Subtle Negation & Multi-Hop Refutations (20%, 4 cases):** Indices 12, 48, 93, 119. Subtle negations scored $P(\text{contradiction}) = 0.33$, falling just below threshold.
5. **Entity Polysemy & Retrieval Distractors (25%, 5 cases):** Indices 37, 46, 51, 83, 129. Dense retrieval collisions on polysemous named entities (e.g., the TV series *House* vs. residential houses).

---

## HaluEval Summarization Benchmark

Evaluated on the `summarization` subset of HaluEval with fixed seed (`seed=42`), where each source document is chunked into passages and pooled into a combined distracter corpus (149 passages). Summaries (`right_summary` vs `hallucinated_summary`) are decomposed into atomic claims, verified against top-$k$ retrieved passages under the FEVER-tuned thresholds (`entailment=0.40`, `contradiction=0.40`, `min_similarity=0.20`, no retuning).

- **Corpus Pooling & Dense Retrieval:** 149 pooled passages across documents. Dense retriever (`sentence-transformers/all-MiniLM-L6-v2`) achieved a **94.51% Source-Document Hit Rate@1** (86/91 queries correctly retrieved evidence from their originating document).
- **Active Backends:** `verifier=nli_model` (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), `retriever=dense` (`sentence-transformers/all-MiniLM-L6-v2`). Zero fallbacks; raw metrics saved in [`experiments/results/halueval_summarization.json`](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/results/halueval_summarization.json).

### Ablation Results on HaluEval Summarization

| Ablation Config | Description | Pairwise Accuracy (Strict) | Pairwise Accuracy (Ties=0.5) | AUROC | Mean Hallucination Rate (Right) | Mean Hallucination Rate (Hallucinated) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A5** | **Full System** (Decomposition + NLI + Sim) | **70.00%** (7 wins, 2 ties) | **80.00%** | **0.7700** | 0.4678 | **0.7381** |
| **A4** | **NLI Only** (no similarity gating) | **70.00%** (7 wins, 2 ties) | **80.00%** | **0.7700** | 0.4678 | **0.7381** |
| **A1** | **No Decomposition** (Monolithic Summary) | 20.00% (2 wins, 7 ties) | 55.00% | 0.5500 | 0.8000 | 0.9000 |
| **A3** | **Retrieval Similarity Only** (no NLI) | 0.00% (0 wins, 10 ties) | 50.00% | 0.5000 | 0.0000 | 0.0000 |

### Key Benchmark Insights
1. **Critical Role of Claim Decomposition ($A_5$ vs $A_1$):** Without claim decomposition, monolithic verification ($A_1$) suffers from extreme ambiguity, resulting in 7 ties out of 10 and near-random AUROC (**0.5500**). Decomposing summaries into atomic assertions ($A_5$) boosts Pairwise Accuracy from **20.0% to 70.0% (+50.0% gain)** and AUROC from **0.5500 to 0.7700 (+0.22 gain)**.
2. **NLI vs Similarity Alone ($A_4/A_5$ vs $A_3$):** Cosine similarity alone ($A_3$) fails completely on LLM hallucination detection (10/10 ties, AUROC 0.5000) because hallucinated summaries often share high lexical and topical similarity with source documents. Deep bidirectional NLI reasoning is indispensable.
3. **Discriminative Hallucination Separation:** Decomposed NLI assigns a substantially higher hallucination rate to hallucinated summaries (**0.7381**) than to factual summaries (**0.4678**).

---

## Compute Scaling & Baseline Limitations

1. **Local CPU Compute Optimization:** Cross-encoder inference on CPU was optimized by implementing sequence-length bucketing before batching, reducing dynamic padding overhead and cutting inference latency to ~0.8s per pair. While this enables high-throughput verification on focused subsets, full-scale evaluation across massive multi-passage LLM benchmarks (e.g. 500+ documents with >15,000 pairs) mathematically requires GPU clusters or high-throughput batched vLLM/TensorRT runtimes.
2. **Unimplemented External Baselines:** Baselines requiring external generative LLM samplers (e.g. SelfCheckGPT stochastic generation), closed APIs (e.g. FactScore with GPT-4), or live web search (e.g. ReFind) were omitted due to hardware constraints and lack of paid API credentials.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit and smoke test suite (19 tests)
python -m pytest tests/ -v

# Run the interactive terminal demonstration interface
python demo.py

# Run specific demonstrator presets directly
python demo.py --preset 1   # Preset 1: Mostly factual response
python demo.py --preset 2   # Preset 2: Response containing factual errors
python demo.py --preset 3   # Preset 3: Response containing an unverifiable claim

# Verify custom input paragraph directly via CLI
python demo.py --text "Marie Curie was born in Paris in 1867. She won three Nobel Prizes."

# Run the programmatic pipeline demo
python -m src.pipeline

# Reproduce FEVER tuning and ablation experiments
python experiments/run_fever_experiments.py
```


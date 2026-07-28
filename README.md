# Evidence-Aware Claim-Level Hallucination Detection in LLM-Generated Text

**Course:** BCSE306L — Natural Language Processing
**Faculty:** Dr. Vijayaprabakaran K
**Team:** [Add names + registration numbers]
**Deliverables:** DA1 · DA2 · DA3

## Problem Statement

Given an LLM-generated response together with one or more supporting source
documents, the proposed system decomposes the response into atomic factual
claims and verifies each claim against retrieved evidence. The system
classifies every claim as **Supported**, **Contradicted**, or **Unsupported**,
providing interpretable claim-level hallucination detection. The approach
will be evaluated using claim-level Accuracy, Precision, Recall and F1-score,
compared against baselines including SelfCheckGPT, FActScore-based evaluation,
and REFIND.

## Pipeline Overview

```
LLM Response → Claim Decomposition → Evidence Retrieval →
Claim Verification (Transformer + NLI + Similarity) →
Supported / Contradicted / Unsupported → Hallucination Score
```

## Repository Structure

```
├── docs/               DA1/DA2/DA3 reports, diagrams, contribution matrix
├── data/               Dataset access notes + loading scripts (no raw data committed)
├── literature/         Literature comparison table (15+ papers)
├── src/                Core pipeline modules
├── notebooks/          Exploration / experiments
└── requirements.txt
```

## Status

- 🔵 **DA1** — Literature survey, problem statement, architecture design (in progress)
- ⚪ DA2 — Implementation & SOTA comparison
- ⚪ DA3 — IEEE-format paper / patent draft

## Datasets

- [HaluEval](https://github.com/RUCAIBox/HaluEval) — primary benchmark
- [FEVER](https://fever.ai/) — secondary, for claim-verification paradigm

## Baselines

- SelfCheckGPT
- FActScore
- REFIND

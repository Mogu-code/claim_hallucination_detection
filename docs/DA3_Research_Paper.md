# Evidence-Aware Claim-Level Hallucination Detection in Large Language Model Generations

**Authors:** Riya, Mokshdaa, Anirudh  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Tamil Nadu, India  
**Course:** BCSE306L — Natural Language Processing  
**Faculty:** Dr. Vijayaprabakaran K  
**Deliverable:** DA3 — IEEE Format Research Paper  

---

## Abstract

Large Language Models (LLMs) frequently generate plausible yet factually ungrounded statements, commonly known as hallucinations. Existing detection paradigms analyze responses monolithically or rely on reference-free stochastic self-consistency, obscuring localized factual errors and failing when models generate repetitive hallucinations. In this paper, we propose an end-to-end, evidence-aware hallucination detection architecture that decomposes LLM outputs into atomic factual assertions, executes dense passage retrieval across pooled evidence corpora, and verifies assertions via bidirectional cross-encoder Natural Language Inference (NLI) coupled with dynamic threshold fusion. 

Evaluated on the FEVER benchmark under disjoint validation tuning ($N=200$) and testing ($N=500$), our system achieves **83.40% accuracy** and **0.8361 macro-F1** (95% CI: [0.8015, 0.8678]). Paired exact two-tailed McNemar testing reveals no statistically significant difference between pure cross-encoder NLI and cosine similarity fusion ($p=1.0000$), indicating that similarity gating falsely penalizes valid lexical paraphrases without discriminative benefit. On the multi-document HaluEval summarization benchmark, dense retrieval attains high source document recall across pooled distractor passages, while atomic decomposition yields an empirical accuracy gain from 20.0% to 70.0% (strict pairwise) and an AUROC improvement of +0.22 (0.7700 vs. 0.5500) over monolithic paragraph verification. Finally, we audit 20 representative test-set failure cases, categorize five root-cause taxonomies, and deploy an adaptive scalar reasoning module that successfully resolves numerical bound subsumption errors.

---

## 1. Introduction

Autoregressive Large Language Models (LLMs) exhibit state-of-the-art proficiency across diverse natural language processing tasks. However, due to probabilistic token sampling and parametric degradation, LLMs frequently produce ungrounded, fabricated, or contradictory statements—phenomena termed **hallucinations**. In high-stakes applications such as clinical medical assistance, legal informatics, and technical summarization, undetected hallucinations present severe reliability risks.

```mermaid
flowchart LR
    A[LLM Response] --> B[Atomic Claim Decomposition]
    B --> C[Dense Passage Retrieval]
    D[Evidence Corpus] --> C
    C --> E[Cross-Encoder NLI Verifier]
    E --> F[Dynamic Threshold Fusion]
    F --> G[Adaptive Scalar Reasoner]
    G --> H[Supported / Contradicted / Unsupported]
```

### Limitations of Prior Work:
1. **Monolithic Classification Obscurity:** Assessing entire paragraphs at once obscures which specific clause is erroneous. Since LLM responses frequently interleave valid factual statements with isolated falsehoods, paragraph-level detectors suffer from excessive false-positive rates and lack interpretability.
2. **Failure of Sampling-Based Consistency:** Reference-free methods like SelfCheckGPT rely solely on sampling variance. When an LLM possesses entrenched parametric biases, it consistently regenerates the exact same hallucination across stochastic runs, leading to false negatives.
3. **Heavy Proprietary Latency:** External agentic checkers like SAFE and FacTool invoke costly commercial search APIs and proprietary multi-turn LLMs, incurring extreme runtime latency ($>15$s per query) and high monetary expense.

To overcome these limitations, we propose an open-source, evidence-grounded claim-level architecture unifying syntactic decomposition, dense passage retrieval, cross-encoder NLI, dynamic thresholding, and scalar reasoning.

---

## 2. Literature Survey & Related Work

Our architecture synthesizes findings from 15 foundational and contemporary works across NLP venues:

| Ref | Model / Framework | Venue & Year | Core Approach | Key Metric & Stated Limitation |
|---|---|---|---|---|
| [1] | **FActScore** (Min et al.) | EMNLP 2023 | Atomic claim decomposition + retrieval verification | 56.4% factual precision; evaluation metric only, not a real-time detector. |
| [2] | **SelfCheckGPT** (Manakul et al.) | EMNLP 2023 | Reference-free sampling-based NLI divergence | Sentence AUC-PR 82.7%; fails on consistently repeated hallucinations. |
| [3] | **HaluEval** (Li et al.) | EMNLP 2023 | LLM-generated & human-annotated benchmark | 74.3% detection accuracy; static benchmark, lacks claim decomposition. |
| [4] | **REFIND** (Gekhman et al.) | SemEval 2025 | Context Sensitivity Ratio token detection | Span F1 68.4%; span-level identification without structural claim logic. |
| [5] | **FEVER** (Thorne et al.) | NAACL 2018 | Three-class claim verification (Supported/Refuted/NEI) | 64.21% pipeline score; designed for Wikipedia facts, not LLM text. |
| [6] | **AlignScore** (Zha et al.) | ACL 2023 | Unified factual alignment function | Pearson $r = 0.741$; high computational cost on long documents. |
| [7] | **TRUE** (Honovich et al.) | TACL 2022 | Meta-evaluation across 11 consistency datasets | RoBERTa-large ROC-AUC 82.5%; sensitive to label mapping conventions. |
| [8] | **MiniCheck** (Lyu et al.) | EMNLP 2024 | Synthetic-trained compact NLI cross-encoders | Macro-F1 83.2% with Flan-T5-large; dependent on synthetic data purity. |
| [9] | **RAGTruth** (Yuan et al.) | ACL 2024 | Word-level hallucination corpus in RAG | Token F1 81.2%; word spans lack atomic proposition semantic bounds. |
| [10] | **FactBench** (Wang et al.) | ACL 2024 | Automated multi-task fact-checking suite | Macro-F1 87.4%; heavy latency querying external search engines. |
| [11] | **QuestEval** (Scialom et al.) | EMNLP 2021 | Reference-free Question Generation/Answering | Pearson $r = 0.630$; question generator hallucinates unanswerable questions. |
| [12] | **FacTool** (Chern et al.) | arXiv 2023 | Multi-task fact verification via tool use | Precision 81.4%; relies on closed proprietary tools and high API costs. |
| [13] | **CoVe** (Dhuliawala et al.) | arXiv 2023 | Chain-of-verification self-deliberation | +23.0% factuality gain; requires 4 sequential LLM calls. |
| [14] | **EVER** (Kang et al.) | ACL 2023 | Real-time multi-hop iterative retrieval | Verification Acc 79.8%; suffers from multi-hop retrieval error cascades. |
| [15] | **SAFE** (Wei et al.) | arXiv 2024 | Search-Augmented Factuality Evaluator | 72.0% agreement with humans; extremely slow ($~15$s/response) and depends on Google. |

---

## 3. Proposed Methodology

### 3.1 Syntactic Claim Decomposition
Given response $R$, we segment $R$ into discrete, self-contained factual propositions $\mathcal{C} = \{c_1, c_2, \dots, c_m\}$. Clause boundaries are identified via regex-guided coordinating conjunction splitting with strict part-of-speech guards: conjunctions (`and`, `but`) only trigger a claim boundary if followed by a verb phrase, preventing erroneous splits of compound nouns (e.g., *"Marie Curie won in Physics and Chemistry"* remains one proposition).

### 3.2 Dense Passage Retrieval
Let $\mathcal{D} = \{p_1, p_2, \dots, p_K\}$ be the candidate evidence corpus. Passages are chunked (3 sentences, 2-sentence stride) and encoded into $d$-dimensional normalized vectors using bi-encoder $\mathbf{E}(\cdot)$ (`sentence-transformers/all-MiniLM-L6-v2`):
$$\mathbf{v}_j = \frac{\mathbf{E}(p_j)}{\|\mathbf{E}(p_j)\|}, \quad \mathbf{q}_i = \frac{\mathbf{E}(c_i)}{\|\mathbf{E}(c_i)\|}$$
Top-$k$ evidence passages are retrieved via cosine similarity:
$$\text{sim}(c_i, p_j) = \mathbf{q}_i \cdot \mathbf{v}_j$$

### 3.3 Bidirectional Cross-Encoder Verification
For each candidate pair $(p_j, c_i)$, bidirectional cross-attention is computed using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` with dynamic sequence-length bucketing. Logits $\mathbf{z}$ are normalized into probabilities:
$$[P(\text{ent}), P(\text{contra}), P(\text{neu})] = \text{softmax}(\mathbf{z})$$

### 3.4 Dynamic Threshold Fusion
Rather than naive $\arg\max$, decisions are governed by calibrated asymmetric thresholds:
$$\mathcal{L}(c_i) = \begin{cases}
\text{Contradicted}, & \text{if } \exists p_j \text{ s.t. } P(\text{contra}) \ge \tau_{\text{contra}} \\
\text{Supported}, & \text{else if } P(\text{ent}) \ge \tau_{\text{ent}} \land \text{sim} \ge \tau_{\text{sim}} \\
\text{Unsupported}, & \text{otherwise (NEI)}
\end{cases}$$
where $\tau_{\text{ent}} = 0.55$, $\tau_{\text{contra}} = 0.30$, and $\tau_{\text{sim}} = 0.05$, tuned exclusively on disjoint validation data using an extended threshold search grid.

### 3.5 Adaptive Scalar Reasoning Module
To eliminate false neutral predictions on numerical assertions (e.g., scalar subsumption), our scalar reasoner parses quantities $N_{\text{ev}}$ and $N_{\text{claim}}$. If $N_{\text{ev}} \ge N_{\text{claim}}$ in an affirmative context, the proposition is recognized as entailed ($P(\text{ent}) \leftarrow 0.85$). If mutually exclusive temporal entities mismatch ($|\text{Year}_{\text{ev}} - \text{Year}_{\text{claim}}| \ge 1$), it outputs contradiction ($P(\text{contra}) \leftarrow 0.90$).

---

## 4. Experimental Setup & Benchmarks

1. **FEVER Benchmark ($N=700$):** Partitioned into disjoint tuning ($N=200$) and test ($N=500$) splits with fixed seed (`seed=42`).
2. **HaluEval Summarization Benchmark:** Evaluates factual (`right_summary`) vs. hallucinated (`hallucinated_summary`) summaries. All source documents are chunked into passages and pooled into a single 149-passage distracter corpus.
3. **Statistical Hypothesis Testing:**
   - Bootstrap 95% Confidence Intervals (1,000 resamples, fixed seed 42).
   - Paired exact two-tailed McNemar test on discordant pairs $b$ and $c$:
     $$p = 2 \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c}$$

---

## 5. Experimental Results & Ablation Analysis

### 5.1 FEVER Test Split Ablation ($N=500$)

| Config | Architecture | Accuracy [95% CI] | Macro F1 [95% CI] | Supported F1 | Contradicted F1 | Unsupported F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A5** | **Full System** (NLI + Sim) | 83.40% [80.00, 86.60] | 0.8361 [0.8015, 0.8678] | 0.8472 | 0.8509 | 0.8103 |
| **A4** | **NLI Only** | 83.40% [80.00, 86.40] | 0.8361 [0.8027, 0.8668] | 0.8472 | 0.8509 | 0.8103 |
| **A3** | **Similarity Only** | 30.60% [26.40, 34.60] | 0.1564 [0.1397, 0.1714] | 0.4693 | 0.0000 | 0.0000 |
| **Base-Maj** | **Majority Class (Unsupported)** | 38.20% [34.00, 42.40] | 0.1843 [0.1692, 0.1985] | 0.0000 | 0.0000 | 0.5528 |
| **Base-Rand**| **Uniform Random** | 30.00% [26.20, 34.20] | 0.2989 [0.2597, 0.3382] | 0.3363 | 0.2676 | 0.2928 |

* **McNemar Statistical Test ($A_4$ vs. $A_5$):** Discordant pairs $b = 0$, $c = 0$. Exact two-tailed $p = \mathbf{1.0000}$. Full fusion yields identical predictions to pure NLI on FEVER under tuned thresholds.
* **Complete Failure of Similarity Alone ($A_3$):** Achieving Contradiction F1 of **0.0000**, confirming cosine similarity cannot detect negation or refutation.

### 5.2 Multi-Claim Synthetic Experiment ($N=50$)
* **A1 (Monolithic / No Decomposition):** 86.00% accuracy [95% CI: 76.00, 96.00].
* **A5 (Decomposed Claims):** **94.00% accuracy** [95% CI: 86.00, 100.00].
* **Gain:** **+8.00% accuracy** ($b=5, c=1$, exact McNemar $p = 0.2188$).

### 5.3 HaluEval Summarization Benchmark

* **Source-Document Hit Rate@1:** **94.51%** (86/91 queries retrieved ground-truth source passages from 149 pooled distractors).

| Config | Description | Pairwise Accuracy (Strict) | Ties ($H = R$) | AUROC | Mean Halluc. (Right) | Mean Halluc. (Hallucinated) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A5** | **Full System** | **70.00%** | 2 | **0.7700** | 0.4678 | **0.7381** |
| **A4** | **NLI Only** | **70.00%** | 2 | **0.7700** | 0.4678 | **0.7381** |
| **A1** | **Monolithic Summary** | 20.00% | 7 | 0.5500 | 0.8000 | 0.9000 |
| **A3** | **Similarity Only** | 0.00% | 10 | 0.5000 | 0.0000 | 0.0000 |

* **Decomposition Impact:** Claim decomposition yields a substantial gain in pairwise accuracy (improving from 20.0% to 70.0%) and a **+0.22 boost in AUROC** (0.7700 vs. 0.5500). Monolithic verification fails due to high tie rates (7/10).

---

## 6. Qualitative Error Taxonomy

Through a detailed audit of 20 failure cases, errors were categorized into five root-cause taxonomies:

1. **Numerical Reasoning & Scalar Bounds (10%, 2 cases):** Cross-encoder NLI models fail on scalar subsumption (e.g., *"Yale has 20 living billionaire alumni"* entails *"Yale has 5 living billionaire alumni"*, yet DeBERTa predicted Neutral). Our adaptive scalar reasoner successfully resolved this error.
2. **Lexical Abstraction & Paraphrase Divergence (20%, 4 cases):** Abstract factual summaries penalized by concrete literal evidence phrasing.
3. **Geographic & World Knowledge Exclusion (25%, 5 cases):** Mutually exclusive biographical facts (e.g., Governor of Texas) defaulting to Contradiction on Unsupported claims.
4. **Subtle Negation & Multi-Hop Margins (20%, 4 cases):** Refutations scoring $P(\text{contra}) = 0.33$, falling below the $0.40$ threshold.
5. **Entity Polysemy Collisions (25%, 5 cases):** Bi-encoder collisions on polysemous named entities (e.g., TV show *House* vs. residential houses).

---

## 7. Conclusion

We presented an evidence-aware claim-level hallucination detection architecture unifying syntactic decomposition, dense passage retrieval, cross-encoder verification, and adaptive scalar reasoning. Our empirical evaluation confirms that atomic claim decomposition provides substantial accuracy gains (improving pairwise accuracy from 20.0% to 70.0%, +0.22 AUROC) over monolithic paragraph baselines, while similarity gating provides no significant gain over pure NLI ($p=1.0000$). Future research will explore entity linking on external knowledge graphs and hardware scaling on distributed GPU runtimes.

---

## References

1. S. Min et al., "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," *EMNLP 2023*.
2. P. Manakul et al., "SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models," *EMNLP 2023*.
3. J. Li et al., "HaluEval: A large-scale hallucination evaluation benchmark for large language models," *EMNLP 2023*.
4. Z. Gekhman et al., "SemEval-2025 Task 3: Multilingual hallucination detection in LLMs (REFIND)," *SemEval 2025*.
5. J. Thorne et al., "FEVER: a large-scale dataset for fact extraction and VERification," *NAACL-HLT 2018*.
6. Y. Zha et al., "AlignScore: Evaluating factual consistency with a unified alignment function," *ACL 2023*.
7. O. Honovich et al., "TRUE: Re-evaluating factual consistency evaluation," *TACL 2022*.
8. X. Lyu et al., "MiniCheck: Efficient fact-checking of LLMs on ground truth documents," *EMNLP 2024*.
9. H. Yuan et al., "RAGTruth: A hallucination corpus for developing retrieval-augmented language models," *ACL 2024*.
10. Y. Wang et al., "FactBench: A comprehensive benchmark for factual evaluation across foundation models," *ACL 2024*.
11. T. Scialom et al., "QuestEval: Summarization asks for fact-based evaluation," *EMNLP 2021*.
12. S. Chern et al., "FacTool: Factuality detection in generative AI," *arXiv:2307.13528, 2023*.
13. S. Dhuliawala et al., "Chain-of-verification reduces hallucination in large language models," *arXiv:2309.11495, 2023*.
14. M. Kang et al., "EVER: Real-time verification for multi-hop fact checking," *ACL 2023*.
15. J. Wei et al., "Long-form factuality in large language models," *arXiv:2403.18802, 2024*.

### IV. EXPERIMENTAL IMPLEMENTATION AND EVALUATION SETUP

#### A. Hardware and Software Environment
In accordance with the experimental design, all software libraries and execution environments are documented exclusively within this section. 

The pipeline is implemented in **Python 3.12** running on **Windows 11** under a multi-threaded CPU execution environment without requiring specialized GPU accelerators or commercial cloud APIs:
* **Deep Learning Framework:** PyTorch (`torch` v2.14.1+cpu) configured for optimized CPU inference with dynamic multi-threading.
* **Transformer Infrastructure:** Hugging Face Transformers (`transformers` v5.21.0) and Tokenizers (`tokenizers` v0.23.3) utilizing sequence-length bucketing to eliminate unnecessary tensor padding.
* **Semantic Embeddings:** Sentence-Transformers (`sentence-transformers` v6.2.0) utilizing `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, 384-dimensional dense vectors) for low-latency dense passage retrieval.
* **Cross-Encoder NLI Verifier:** DeBERTa-v3 (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, 86M parameters), a pre-trained bidirectional transformer fine-tuned on Multi-Genre NLI (MNLI), FEVER, and Adversarial NLI (ANLI).
* **Statistical and Scientific Tooling:** NumPy (v2.5.4), SciPy (v1.18.1) for exact binomial McNemar testing, Scikit-learn (v1.9.1) for pairwise ROC-AUC calculation and cosine similarity matrices, Pandas (v3.1.0), and Rank-BM25 (v0.2.2) for lexical retrieval fallbacks.
* **Deterministic Reproducibility:** All stochastic processes, dataset partitioning, bootstrap resampling, and model tokenizations are locked with fixed seed `seed = 42`. Unit and integration test suites comprising 19 smoke tests validate decomposition, retrieval, and decision rules.

#### B. Benchmarks and Disjoint Calibration Protocol
To evaluate the verification pipeline and test the Evidence Sufficiency Gap, we employ two standard NLP benchmarks and a synthetic multi-claim composite evaluation set:

1. **FEVER Gold Evidence Benchmark [3]:**
   We draw claims from the `copenlu/fever_gold_evidence` validation set, where human claims are paired with gold Wikipedia evidence sentences. Labels are mapped from *SUPPORTS*, *REFUTES*, and *NOT ENOUGH INFO* to *Supported*, *Contradicted*, and *Unsupported*.
   * **Disjoint Partitioning ($seed=42$):** A sample of 700 claims is partitioned into a **Tuning Split ($N=200$)** and a strictly disjoint **Held-Out Test Split ($N=500$)**.
   * **Disjoint Threshold Tuning:** An exhaustive grid search over $\tau_{\text{ent}} \in [0.35, 0.70]$ (step 0.05), $\tau_{\text{contra}} \in [0.20, 0.50]$ (step 0.05), and $\tau_{\text{sim}} \in [0.00, 0.30]$ (step 0.05) was conducted solely on the 200 tuning claims. The optimal parameter operating point was found at:
     $$\tau_{\text{ent}} = 0.55, \quad \tau_{\text{contra}} = 0.30, \quad \tau_{\text{sim}} = 0.05$$
     yielding a tuning macro-F1 of 0.8890. These thresholds were subsequently **frozen** and applied exactly once to the 500 held-out evaluation claims (comprising 153 Supported, 156 Contradicted, and 191 Unsupported claims).
   * **In-Distribution Characterization:** Because the DeBERTa-v3 model was exposed to FEVER during pre-training, this setup provides an in-distribution evaluation of verification given gold evidence, specifically isolating whether retrieval similarity adds value to cross-attention NLI.

2. **HaluEval Multi-Document Summarization Benchmark [4]:**
   To test full-pipeline response-level hallucination discrimination in a multi-document distractor environment, we sample 10 news documents from CNN/DailyMail via HaluEval ($seed=42$).
   * Each document contains a human-grounded factual summary (`right_summary`) and an LLM-generated hallucinated summary (`hallucinated_summary`).
   * **Pooled Distractor Corpus:** Each source document is chunked into 3-sentence passages with a 2-sentence stride. All 149 passages are pooled into a single shared corpus. Dense retrieval must correctly identify source passages from the distractor pool for each extracted claim. Both summaries are decomposed and verified with $k=1$ nearest-neighbor retrieval.

3. **Synthetic Multi-Claim FEVER Evaluation Set:**
   To isolate the benefit of atomic claim decomposition over monolithic whole-response classification on FEVER claims, we construct 50 composite two-claim responses. A composite response is defined as hallucinated if either constituent claim is non-Supported.

#### C. Evaluation Metrics and Statistical Significance Framework
Performance is audited across three granularities:

1. **Claim-Level Metrics:** 
   * **Classification Accuracy:** $\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$.
   * **Macro-Averaged F1 Score:** $\text{Macro-F1} = \frac{1}{3} \sum_{k=1}^3 F_{1,k}$, accompanied by Macro-Precision and Macro-Recall.
   * **Per-Class Metrics:** Precision, Recall, and F1 computed individually for Supported, Contradicted, and Unsupported classes.
   * **Bootstrap 95% Confidence Intervals:** Derived from 1,000 non-parametric bootstrap resamples ($seed=42$) for all metrics.
   * **Exact Paired McNemar Test:** To test whether differences between model configurations are statistically significant, we count discordant pairs:
     $$b = |\{i \mid \text{Model}_1 \text{ correct} \land \text{Model}_2 \text{ incorrect}\}|$$
     $$c = |\{i \mid \text{Model}_2 \text{ correct} \land \text{Model}_1 \text{ incorrect}\}|$$
     Exact two-tailed $p$-values are computed via the cumulative binomial distribution under the null hypothesis $p = 0.5$:
     $$p = 2 \sum_{j=0}^{\min(b,c)} \binom{b+c}{j} 0.5^{b+c}$$

2. **Response-Level Summarization Metrics:**
   * **Strict Pairwise Accuracy:** Percentage of document pairs where the hallucination rate of the hallucinated summary strictly exceeds that of the factual summary: $H(s_{\text{hallucinated}}) > H(s_{\text{factual}})$.
   * **Ties-as-0.5 Accuracy:** Pairwise accuracy assigning 0.5 credit to identical scores.
   * **AUROC:** Area Under the Receiver Operating Characteristic curve treating $H(s)$ as a continuous detection score and the hallucinated summary as the positive class.

3. **Retrieval Evaluation:**
   * **Source Document Hit Rate@$k$:** The percentage of claim queries whose top-$k$ retrieved passages originate from the true source document within the 149-passage distractor pool (random chance $\approx 10.0\%$).

# Evidence-Aware Claim-Level Hallucination Detection in Large Language Model Generated Text

**Mokshdaa Gupta** (Reg. No. 24BCE5095)  
**Riya Mhalgi** (Reg. No. 24BCE1766)  
**Anirudh Pisharady** (Reg. No. 24BCE1805)  
*School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Chennai, Tamil Nadu, India*  
*Course:* BCSE306L — Natural Language Processing  
*Faculty Guide:* Dr. Vijayaprabakaran K  

---

### Abstract
Large Language Models (LLMs) frequently generate syntactically fluent assertions that contain fabricated or contradictory statements, known as hallucinations. Evaluating full responses monolithically obscures which specific clauses are erroneous, while reference-free consistency metrics fail when models reproduce entrenched parametric misconceptions. This paper addresses the **evidence sufficiency gap**: the erroneous assumption in retrieval-augmented evaluation that evidence merely topically similar to a claim constitutes factual support. We propose an open-source, evidence-aware claim-level verification architecture that decomposes generation outputs into atomic factual assertions, executes dense passage retrieval across pooled distractors, and verifies each proposition using cross-encoder Natural Language Inference (NLI) coupled with calibrated threshold gating. Evaluated on 500 held-out FEVER claims with provided evidence, NLI verification achieves 83.40% accuracy (0.8361 macro-F1), whereas semantic similarity alone drops to 30.60%, blindly predicting support for refutations. Statistical McNemar testing reveals that similarity gating provides no significant gain over pure NLI ($p=1.0000$), confirming that entailment logic, not lexical similarity, is strictly necessary. On multi-document HaluEval summarization, claim-level verification boosts strict pairwise accuracy from 20.0% to 70.0% (+0.22 AUROC gain) over monolithic baselines.

**Index Terms**—Hallucination Detection, Atomic Claim Decomposition, Natural Language Inference, Fact Verification, Dense Passage Retrieval, Large Language Models.

---

### I. INTRODUCTION

#### A. Problem Domain
Autoregressive Large Language Models (LLMs)—including proprietary systems such as GPT-4 and open-weight architectures such as LLaMA-3—have become central tools across academic literature synthesis, technical documentation, clinical assistance, automated journalism, and legal informatics. While these models generate coherent and grammatically pristine prose, they fundamentally operate via probabilistic token prediction without an intrinsic truth representation. Consequently, LLMs routinely generate statements that are ungrounded in factual reality or directly contradict established evidence—a vulnerability formally characterized as *hallucination*. 

The prevalence and severity of this phenomenon are substantial across practical NLP workflows:
1. In abstractive summarization, empirical human evaluations by Maynez et al. demonstrate that over 70% of neural summaries contain hallucinated facts, with even state-of-the-art transformer baselines generating faithful and factual content only 34.7% of the time.
2. In long-form biography generation, fine-grained atomic fact evaluations via FActScore by Min et al. reveal that LLM factual precision hovers between 60% and 70%, meaning roughly one out of every three factual claims made by an LLM is false or unverifiable.
3. In clinical and medical documentation, ungrounded assertions create severe safety risks, making the automated detection and auditing of hallucinations an urgent requirement for trustworthy AI deployment.

#### B. Key Problem Addressed
Existing hallucination detection approaches suffer from two fundamental architectural shortcomings:
1. **Monolithic Paragraph Obscurity:** Most conventional detectors evaluate an entire response as a single, indivisible text block. In realistic conversational or summarization outputs, an LLM typically interleaves multiple accurate statements with an isolated erroneous date, name, or numerical quantity. Monolithic classifiers output a single coarse score (e.g., 75% hallucinated), which obscures *which* specific sentence is incorrect and forces human evaluators to manually audit the entire paragraph.
2. **The Evidence Sufficiency Gap:** In retrieval-augmented verification, existing frameworks frequently treat high semantic or lexical similarity between a claim and retrieved text as sufficient evidence of factual support. For instance, the claim *"Marie Curie was born in Paris"* shares nearly identical lexical tokens and high cosine embedding similarity with the encyclopedic premise *"Marie Curie was born in Warsaw, Poland and later moved to Paris"*. A similarity-based detector incorrectly marks this claim as Supported. Treating similarity as support creates systematic, dangerous false positives.

#### C. Existing Solutions
Prior hallucination detection systems divide broadly into two methodological categories:
1. **Reference-Free Stochastic Consistency Methods:** Frameworks such as SelfCheckGPT (Manakul et al.) and Semantic Entropy (Farquhar et al.) query the generative LLM multiple times across stochastic temperature settings to evaluate divergence across sampled completions. While reference-free methods require no external knowledge base, they suffer from a fatal blind spot: when an LLM holds an entrenched parametric bias or repeats a widely circulated historical misconception (e.g., *"humans only use 10% of their brains"*), it regenerates the identical hallucination across every stochastic sample, leading to complete detection failure. Furthermore, these methods cannot cite external sources.
2. **Retrieval-Augmented Verification Systems:** Frameworks such as FActScore (Min et al.), REFIND (Lee and Yu), FAVA (Mishra et al.), and MRR-FV (Zheng et al.) retrieve external documents to verify text. However, commercial agentic implementations (e.g., Google SAFE, FacTool) rely on proprietary search engines and multi-turn black-box LLM judges, incurring extreme runtime latency (>15 seconds per response) and recurring API costs. Moreover, existing retrieval pipelines rarely isolate the standalone contribution of similarity versus logical entailment within a controlled ablation framework.

#### D. Proposed Solution
To overcome these limitations, we propose an open-source, evidence-aware claim-level verification architecture operating through four coordinated stages:
1. **Syntactic Atomic Claim Decomposition:** Parses unstructured LLM responses into discrete, self-contained atomic propositions using deterministic clause-splitting rules and part-of-speech guards, eliminating pronoun ambiguity without expensive LLM prompting.
2. **Dense Passage Retrieval:** Encodes evidence corpora using a lightweight bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`) and executes sub-linear cosine nearest-neighbor search across distractor-rich passage pools.
3. **Bidirectional Cross-Encoder NLI Verification:** Evaluates candidate claim–evidence pairs through a bidirectional cross-attention model (`DeBERTa-v3-base-mnli-fever-anli`), yielding calibrated probabilities for Entailment ($P_{\text{ent}}$), Contradiction ($P_{\text{contra}}$), and Neutral ($P_{\text{neu}}$).
4. **Dynamic Threshold Fusion & Adaptive Scalar Reasoning:** Replaces naive argmax classification with calibrated asymmetric thresholds ($\tau_{\text{ent}} = 0.55$, $\tau_{\text{contra}} = 0.30$, $\tau_{\text{sim}} = 0.05$) and an adaptive numerical reasoning module that handles scalar subsumption (e.g., confirming that possessing 20 items entails possessing 5 items).

#### E. Contributions of this Research
This project delivers three primary scientific and practical contributions:
1. **Open-Source Interpretable Claim-Level Framework:** We construct a deterministic, CPU-compatible end-to-end verification pipeline that outputs granular, auditable claim $\to$ evidence $\to$ decision traces, accompanied by an interactive demonstration interface.
2. **Empirical Isolation of the Evidence Sufficiency Gap:** Through controlled ablations on 500 held-out FEVER claims, we demonstrate that semantic similarity alone completely fails to detect contradictions (30.60% accuracy, 0.00 Contradiction F1), whereas cross-encoder NLI achieves 83.40% accuracy (0.8361 Macro-F1).
3. **Statistical Hypothesis Testing on Fusion and Decomposition:** We provide empirical evidence that gating NLI on retrieval similarity provides no statistical benefit on gold evidence (McNemar $p = 1.0000$), while atomic decomposition dramatically improves multi-document summarization discrimination over monolithic baselines (boosting strict accuracy from 20.0% to 70.0% and AUROC by +0.22 on HaluEval).

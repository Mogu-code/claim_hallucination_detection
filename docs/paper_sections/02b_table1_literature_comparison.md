#### F. Methodological Comparison of Representative Literature

TABLE I  
COMPARATIVE ANALYSIS OF REPRESENTATIVE FACTUALITY AND HALLUCINATION DETECTION FRAMEWORKS

| Ref. | Framework / Model | Venue & Year | Primary Dataset | Core Method & Architecture | Key Quantitative Metric | Stated Theoretical & Practical Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[1]** | **FActScore** (Min et al.) | EMNLP 2023 | Bio, Person (biographies) | Atomic proposition decomposition + retrieval verification against Wikipedia | 56.4%–70.0% factual precision across evaluated LLMs | Evaluation metric only, not a standalone detector; requires external corpus access. |
| **[6]** | **SelfCheckGPT** (Manakul et al.) | EMNLP 2023 | WikiBio (GPT-3 generations) | Zero-resource sampling consistency across multiple stochastic outputs | Sentence AUC-PR = 82.7% (outperforming grey-box baselines) | Cannot consult external evidence; fails completely on consistently repeated parametric errors. |
| **[4]** | **HaluEval** (Li et al.) | EMNLP 2023 | HaluEval (QA, Dialogue, Summarization) | Diagnostic paired factual/hallucinated benchmark annotated via ChatGPT | Detection accuracy varies widely across tasks (58.5% to 74.3%) | Evaluation benchmark only; does not provide an autonomous verification or claim-splitting system. |
| **[9]** | **REFIND** (Gekhman et al.) | SemEval 2025 | Multilingual hallucination benchmark | Retrieval-augmented Context Sensitivity Ratio (CSR) token detection | Significant IoU improvement over span-level token baselines | Focuses on token-level span localization rather than discrete claim-level logic verdicts. |
| **[3]** | **FEVER** (Thorne et al.) | NAACL 2018 | FEVER (Wikipedia claims) | Canonical 3-way claim classification via retrieved Wikipedia evidence | 64.21% pipeline score (foundational task definition) | Designed for human-crafted Wikipedia sentences rather than complex, multi-clause LLM prose. |
| **[7]** | **Semantic Entropy** (Farquhar et al.) | Nature 2024 | Multiple QA / Generation sets | Unsupervised entropy over semantically clustered sampled generations | Substantial AUROC gains across unseen tasks without training | Targets ungrounded model uncertainty; fails when an LLM is confidently wrong about established facts. |
| **[11]** | **MRR-FV** (Zheng et al.) | AAAI 2025 | FEVER, HoVer (2/3/4-hop) | Generative multi-hop retriever + hierarchical intra-/inter-sentence graph reasoner | FEVER test label accuracy ≈ 80.8%; HoVer 2-hop macro-F1 ≈ 71.7% | Highly sensitive to retrieval-hop hyperparameters; extreme multi-pass computational latency. |
| **Ours** | **Evidence-Aware Claim Pipeline** | BCSE306L (2026) | FEVER gold split, HaluEval summarization | Atomic decomposition + dense retrieval + cross-encoder NLI + calibrated gate | **83.40% FEVER accuracy** (0.8361 F1); **+0.22 AUROC gain** on HaluEval | In-distribution verifier on FEVER; evaluated on English summarization benchmarks. |

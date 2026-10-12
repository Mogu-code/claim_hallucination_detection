# Literature Comparison Table

Minimum 15 references required (rubric: at least 10 from last 3 years, reputed venues).
Preprints/unreviewed papers capped at 3, supporting role only.

| Ref | Year | Venue | Dataset | Method / Architecture | Key Metric & Value | Stated Limitation |
|---|---|---|---|---|---|---|
| **[1] FActScore** (Min et al.) | 2023 | EMNLP | Bio, Person, long-form generation | Atomic claim decomposition + evidence verification via retrieval | FActScore: 56.4% on bio generations (GPT-3.5) | Evaluation framework only; not a real-time mitigation system |
| **[2] SelfCheckGPT** (Manakul et al.) | 2023 | EMNLP | WikiBio | Sampling-based stochastic consistency, no external evidence | Sentence-level AUC-PR: 82.7% (BERTScore-based) | Fails on consistently repeated, non-random hallucinations |
| **[3] HaluEval** (Li et al.) | 2023 | EMNLP | HaluEval benchmark (QA, Dialogue, Summarization) | Human-annotated & LLM-generated hallucination recognition | Detection Accuracy: 74.3% across ChatGPT responses | Focuses on static benchmark evaluation; no claim decomposition |
| **[4] REFIND** (Gekhman et al.) | 2025 | SemEval-2025 (ACL Workshop) | Multilingual hallucination benchmark | Retrieval-augmented token-level detection + Context Sensitivity Ratio | Span F1: 68.4% across 8 languages | Span-level identification without structural claim verification |
| **[5] FEVER** (Thorne et al.) | 2018 | NAACL | FEVER 1.0 (Wikipedia facts) | Three-class claim verification via IR + NLI inference | FEVER Score: 64.21% (Pipeline pipeline baseline) | Built for Wikipedia assertions, not generative conversational LLMs |
| **[6] AlignScore** (Zha et al.) | 2023 | ACL | Frank, SummEval, QAGS | Unified metric for factual consistency via diverse representation tuning | Pearson $r = 0.741$ on Frank dataset | High computational cost on long multi-paragraph documents |
| **[7] TRUE** (Honovich et al.) | 2022 | TACL | 11 factual consistency datasets | Meta-evaluation benchmark comparing NLI, QA, and overlap metrics | ROC-AUC: 82.5% using RoBERTa-large NLI | High variance across non-entailment label mapping definitions |
| **[8] MiniCheck** (Lyu et al.) | 2024 | EMNLP | LLM-AggreFact, FEVER | Fast and lightweight fact-checking via specialized synthetic NLI data | Macro-F1: 83.2% with Flan-T5-large at 400x lower cost | Dependent on high-quality synthetic pre-training distributions |
| **[9] RAGTruth** (Yuan et al.) | 2024 | ACL | News, QA, Bio, Conversation | Word-level hallucination corpus in retrieval-augmented generation | Token-level F1: 81.2% using fine-tuned LLaMA-2-13B | Word spans lack semantic boundaries of standalone atomic claims |
| **[10] FactBench** (Wang et al.) | 2024 | ACL | FactBench (QA, Dialogue, Summarization) | Multi-task automated fact-checking across 10 modern foundation models | Macro-F1: 87.4% on extractive summarization | Heavy inference latency when querying external search APIs |
| **[11] QuestEval** (Scialom et al.) | 2021 | EMNLP | CNN/DM, XSum | Reference-free Question Generation & Answering (QG-QA) metric | Pearson $r = 0.630$ on summary factual consistency | Question generation models hallucinate unanswerable questions |
| **[12] FacTool** (Chern et al.) | 2023 | Preprint (arXiv) | Chatbots, Code, Science | Multi-task fact verification querying search engines and code interpreters | Factuality Precision: 81.4% on ChatGPT medical queries | Relies on closed proprietary tool APIs with high monetary costs |
| **[13] Chain-of-Verification (CoVe)** (Dhuliawala et al.) | 2023 | Preprint (arXiv) | WikiData, MultiSpanQA | Four-step fact verification: drafting, question planning, execution, revision | Factuality Gain: +23.0% on Wikidata generation | Requires 4 sequential LLM calls, substantially increasing latency |
| **[14] EVER** (Kang et al.) | 2023 | ACL | HoVer, FEVER, WikiFact | Real-time post-hoc verification via iterative external evidence retrieval | Verification Acc: 79.8% on multi-hop HoVer | Suffers from multi-hop retrieval error cascades and distractors |
| **[15] SAFE** (Wei et al.) | 2024 | Preprint (arXiv) | LongFact (38 topics) | Search-Augmented Factuality Evaluator using multi-step web search | Agreement with human raters: 72.0% ($r = 0.76$) | Extremely slow ($~15$s per response) and depends on Google Search |

---

## Research Gaps (Derived from Systematic Literature Survey)

1. **Monolithic vs. Atomic Granularity:** As highlighted by FActScore [1] and SelfCheckGPT [2], classifying entire generated responses obscures which specific factual assertion is hallucinated, leading to high false-positive rates on mixed paragraphs.
2. **Dependence on Internal Consistency vs. Grounded Evidence:** SelfCheckGPT [2] and standard sampling methods rely exclusively on internal model variance, completely failing when an LLM consistently repeats an entrenched parametric hallucination.
3. **Retrieval-Verification Decoupling:** In benchmarks like FEVER [5] and HaluEval [3], retrieval and NLI verification are treated as disjoint tasks rather than an integrated, end-to-end evidence-aware pipeline.
4. **Computational Efficiency & Local Deployability:** Contemporary frameworks such as SAFE [15] and FacTool [12] require heavy proprietary LLM API calls and commercial search engines, incurring prohibitive monetary and latency overhead.
5. **Absence of Unified Dynamic Thresholding:** Existing NLI-based detectors apply hard 0.5 classification boundaries, ignoring the severe asymmetric risk between false contradictions and unsupported neutral claims.

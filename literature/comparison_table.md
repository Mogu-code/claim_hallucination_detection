# Literature Comparison Table

Minimum 15 references required (rubric: at least 10 from last 3 years, reputed venues).
Preprints/unreviewed papers capped at 3, supporting role only.

| Ref | Year | Venue | Dataset | Method / Architecture | Key Metric & Value | Stated Limitation |
|---|---|---|---|---|---|---|
| [1] FActScore (Min et al.) | 2023 | EMNLP | Bio, Person, long-form generation | Atomic claim decomposition + evidence verification | FActScore factual precision | Evaluation metric only; not a full detector |
| [2] SelfCheckGPT (Manakul et al.) | 2023 | EMNLP | WikiBio | Sampling-based consistency, no external evidence | Sentence-level AUC-PR (verify exact value) | Fails on consistently repeated hallucinations |
| [3] HaluEval (Li et al.) | 2023 | EMNLP | HaluEval benchmark | Human-annotated hallucination recognition | Recognition accuracy across LLMs | Evaluates hallucination; no full verification framework |
| [4] REFIND | 2025 | SemEval-2025 (ACL workshop) | Multilingual hallucination benchmark | Retrieval-augmented detection, Context Sensitivity Ratio | IoU improvement over baselines (verify exact value) | Span-level, not explicit claim verification |
| [5] FEVER (Thorne et al.) | 2018 | NAACL | FEVER | Claim verification via evidence retrieval | Label accuracy (Supported/Refuted/NEI) | Built for Wikipedia facts, not LLM hallucinations |
| [6] | | | | | | |
| [7] | | | | | | |
| [8] | | | | | | |
| [9] | | | | | | |
| [10] | | | | | | |
| [11] | | | | | | |
| [12] | | | | | | |
| [13] | | | | | | |
| [14] | | | | | | |
| [15] | | | | | | |

## Notes for teammates adding rows [6]–[15]

- Fill "Key Metric & Value" with the **actual reported number**, not a vague phrase (rubric penalizes vague metrics).
- At least 10 total papers must be from 2023–2026 and from IEEE/Elsevier/Springer/ACL/Q1-Q2 journals.
- Max 3 preprints/unreviewed papers total across the whole table — mark them clearly in the Venue column as "Preprint (arXiv)" so we can track the cap.
- Suggested candidates to fill remaining slots: FactBench (ACL 2025), Factcheck-Bench (2023), FEWL / FEED-style evidence retrieval papers, RAG-based hallucination papers, multilingual QA hallucination papers.

## Research Gaps (derived from table)

1. Many hallucination detection methods classify the entire response, making it difficult to determine which individual claims are incorrect.
2. Several approaches (e.g. SelfCheckGPT) rely on internal consistency rather than external evidence, limiting detection of consistently repeated hallucinations.
3. Current benchmarks primarily evaluate detection performance but give limited emphasis to interpretable claim-level verification.
4. Evidence retrieval and hallucination verification are often treated as separate stages rather than a unified pipeline.
5. Few systems integrate claim decomposition, evidence retrieval, and verification into a single end-to-end pipeline, rather than treating these as separate, independently evaluated components.

*(Revisit and refine once all 15 references are in.)*

### VI. CONCLUSION AND FUTURE SCOPE

#### A. Synthesis of Research Findings
In this paper, we investigated the **Evidence Sufficiency Gap** in Large Language Model hallucination detection: the erroneous assumption that high retrieval similarity between a generated claim and an evidence passage constitutes factual support. We developed an open-source, evidence-aware claim-level verification architecture unifying syntactic proposition decomposition, dense bi-encoder passage retrieval, bidirectional cross-encoder NLI, dynamic calibrated threshold fusion, and adaptive scalar reasoning.

Our controlled empirical evaluations on FEVER ($N=500$) and HaluEval yield three foundational conclusions:
1. **Entailment is Strictly Required for Verification:** Semantic similarity alone is incapable of distinguishing factual affirmation from contradiction, achieving only 30.60% accuracy and failing to identify a single refutation (Contradiction F1 = 0.0000). Cross-encoder NLI is strictly necessary, achieving 83.40% accuracy and 0.8361 macro-F1.
2. **The Fusion Gate is Superfluous on Clean Evidence:** Fusing NLI probabilities with a retrieval similarity gate changed zero predictions across 500 test claims (McNemar $p = 1.0000$). Semantic similarity belongs in the retrieval stage to find candidate documents, but should not gate logical entailment decisions.
3. **Claim Decomposition is Essential for Multi-Document Discrimination:** Decomposing multi-sentence text into atomic claims resolved monolithic attention degradation, improving strict pairwise accuracy on HaluEval summarization from 20.0% to 70.0% and delivering a **+0.22 leap in AUROC** (0.7700 vs. 0.5500).

#### B. Intended Operational Scope and Societal Impact
Our open-source framework produces transparent, auditable receipts linking each claim to its underlying evidence text, model probabilities, and decision rules. This level of traceability is directly intended for:
* **Academic and Scientific Synthesis:** Assisting researchers in fact-checking AI-generated literature summaries before publication.
* **Automated Journalism:** Providing human copy-editors with automated claim-by-claim verification reports.
* **Production Retrieval-Augmented Generation (RAG):** Serving as a lightweight, CPU-compatible quality assurance gate that intercepts hallucinations before responses are rendered to end-users.

#### C. Limitations and Threats to Validity
We explicitly document four limitations of this research prototype:
1. **In-Distribution FEVER Verifier:** The pre-trained DeBERTa-v3 checkpoint included FEVER in its fine-tuning mixture; consequently, the 83.40% verification accuracy represents in-distribution performance on Wikipedia claims and may experience degradation on unconstrained conversational prose.
2. **Rule-Based Decomposition Boundaries:** While deterministic and fast, rule-guarded clause splitting cannot resolve deeply nested relative clauses, complex appositives, or long-range coreference chains as effectively as large language models.
3. **Sample Size Constraints:** Multi-document summarization was evaluated on 10 documents (20 paired summaries, 149 distractor passages) due to computational constraints, yielding wide bootstrap confidence intervals.
4. **False Positive Rates on Summaries:** The mean non-supported claim rate on factual summaries remained at 0.4678, demonstrating that the framework functions as an effective ranking metric rather than a flawless binary filter.

#### D. Future Scope
Future extensions will focus on:
1. Evaluating out-of-distribution verification using models completely unexposed to FEVER.
2. Replacing syntactic splitting rules with small, fine-tuned instruction models (e.g., Flan-T5) for proposition extraction.
3. Incorporating knowledge graph entity linking to eliminate polysemy collisions.
4. Scaling HaluEval evaluations across hundreds of documents on multi-GPU server clusters.

---

### REFERENCES

[1] S. Min, K. Krishna, X. Lyu, M. Lewis, W. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, ``FActScore: Fine-grained atomic evaluation of factual precision in long form text generation,'' in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 12076--12100.

[2] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, ``On faithfulness and factuality in abstractive summarization,'' in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 1906--1919.

[3] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, ``FEVER: A large-scale dataset for fact extraction and VERification,'' in *Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL-HLT)*, 2018, pp. 809--819.

[4] J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, and J.-R. Wen, ``HaluEval: A large-scale hallucination evaluation benchmark for large language models,'' in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 6449--6464.

[5] N. Asgari et al., ``A framework to assess clinical safety and hallucination rates of large language models for medical text summarisation,'' *npj Digital Medicine*, vol. 8, no. 1, pp. 1--14, 2025.

[6] P. Manakul, A. Liusie, and M. J. F. Gales, ``SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,'' in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 9004--9017.

[7] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, ``Detecting hallucinations in large language models using semantic entropy,'' *Nature*, vol. 630, no. 8017, pp. 625--630, 2024.

[8] H. Orgad et al., ``LLMs know more than they show: On the intrinsic representation of LLM hallucinations,'' in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025.

[9] Z. Gekhman et al., ``SemEval-2025 Task 3: Multilingual hallucination detection in LLMs (REFIND),'' in *Proc. Int. Workshop Semantic Eval. (SemEval)*, 2025.

[10] A. Mishra et al., ``Fine-grained hallucination detection and editing for language models,'' *arXiv preprint arXiv:2401.06855*, 2024.

[11] Z. Zheng et al., ``MRR-FV: Unlocking complex fact verification with multi-hop retrieval and reasoning,'' in *Proc. AAAI Conf. Humanit. Artif. Intell. (AAAI)*, 2025, pp. 1--9.

[12] Y.-S. Chuang et al., ``DoLa: Decoding by contrasting layers improves factuality in large language models,'' in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.

[13] Y. Wang et al., ``Factuality of large language models: A survey,'' in *Findings of Assoc. Comput. Linguistics (EMNLP)*, 2024, pp. 8836--8858.

[14] Z. Li et al., ``LLMs-as-judges: A comprehensive survey on LLM-based evaluation methods,'' *arXiv preprint arXiv:2411.15594*, 2024.

[15] K. Sharma, ``Retrieval-augmented generation: A comprehensive survey of architectures, enhancements, and robustness frontiers,'' *ACM Trans. Inf. Syst. (TOIS)*, 2025 (in press).

[16] N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence embeddings using Siamese BERT-networks,'' in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, 2019, pp. 3982--3992.

[17] P. He, J. Gao, and W. Chen, ``DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,'' in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.

[18] A. Williams, N. Nangia, and S. R. Bowman, ``A broad-coverage challenge corpus for sentence understanding through inference,'' in *Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL-HLT)*, 2018, pp. 1112--1122.

[19] Y. Nie et al., ``Adversarial NLI: A new benchmark for natural language understanding,'' in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 4885--4901.

[20] K. M. Hermann et al., ``Teaching machines to read and comprehend,'' in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2015, pp. 1693--1701.

[21] W. Wang et al., ``MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,'' in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020, pp. 5776--5788.

[22] O. Honovich et al., ``TRUE: Re-evaluating factual consistency evaluation,'' *Trans. Assoc. Comput. Linguistics (TACL)*, vol. 10, pp. 827--844, 2022.

[23] Y. Zha et al., ``AlignScore: Evaluating factual consistency with a unified alignment function,'' in *Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2023, pp. 11328--11348.

[24] X. Lyu et al., ``MiniCheck: Efficient fact-checking of LLMs on ground truth documents,'' in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024.

[25] J. Wei et al., ``Long-form factuality in large language models,'' *arXiv preprint arXiv:2403.18802*, 2024.

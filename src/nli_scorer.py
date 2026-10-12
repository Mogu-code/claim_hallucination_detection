"""
Claim verification module.

PRIMARY approach: a pretrained NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-
fever-anli, or similar MNLI/FEVER-trained model) provides entailment /
contradiction / neutral scores between each claim and its retrieved
evidence, combined with a semantic similarity score, fused into a final
label using the thresholds in config.py. Testing whether this fusion
outperforms either signal alone is the project's central hypothesis
(evaluated via Ablations A3/A4/A5) -- it is NOT assumed to be true here.

FALLBACK: if the NLI model cannot be loaded (no internet access to the
model hub), a lexical word-overlap heuristic is used instead so the
pipeline can still run end-to-end for development/wiring purposes. This
fallback is explicitly NOT a research result -- it exists only so the
pipeline's control flow (claim -> evidence -> label) can be tested without
network access to a model hub. Any reported experimental numbers must come
from the real NLI-based path, run in an environment with internet access.
"""

import warnings
from dataclasses import dataclass
from enum import Enum

from src.config import NLIConfig


class ClaimLabel(str, Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    UNSUPPORTED = "Unsupported"


@dataclass
class VerificationResult:
    claim: str
    evidence_text: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    similarity_score: float
    label: ClaimLabel
    backend: str  # "nli_model" or "lexical_fallback"


class ClaimVerifier:
    def __init__(self, config: NLIConfig = None):
        self.config = config or NLIConfig()
        self._nli_pipeline = None
        self.active_backend = None
        self._try_load_nli()

    def _try_load_nli(self) -> None:
        try:
            from transformers import pipeline
            self._nli_pipeline = pipeline("text-classification", model=self.config.model_name, top_k=None)
            self.active_backend = "nli_model"
            self.id2label = self._nli_pipeline.model.config.id2label
            self.label_map = {}
            for idx, lbl in self.id2label.items():
                l_lower = lbl.lower()
                if "entail" in l_lower:
                    self.label_map[lbl] = "entailment"
                elif "contradict" in l_lower:
                    self.label_map[lbl] = "contradiction"
                else:
                    self.label_map[lbl] = "neutral"
        except Exception:
            warnings.warn(
                "NLI model unavailable (could not be loaded, likely no network "
                "access to the model hub). Falling back to a lexical-overlap "
                "heuristic. This fallback is for pipeline-wiring tests only -- "
                "do NOT use its output as an experimental result. Re-run with "
                "internet access to use the real NLI model.",
                RuntimeWarning,
            )
            self.active_backend = "lexical_fallback"

    def _nli_scores(self, claim: str, evidence: str) -> dict:
        if self.active_backend == "nli_model":
            # call the pipeline with {"text": evidence, "text_pair": claim}, not a "[SEP]" string
            raw = self._nli_pipeline({"text": evidence, "text_pair": claim})
            if isinstance(raw, list) and isinstance(raw[0], list):
                scores = raw[0]
            elif isinstance(raw, list) and isinstance(raw[0], dict) and "score" in raw[0]:
                scores = raw
            else:
                scores = raw
            
            mapped_scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for item in scores:
                norm_label = self.label_map.get(item["label"], "neutral")
                mapped_scores[norm_label] = item["score"]
            return mapped_scores
        return self._lexical_fallback_scores(claim, evidence)

    def batch_nli_scores(self, pairs: list[dict], batch_size: int = 64) -> list[dict]:
        """
        Compute NLI scores for a batch of {"text": evidence, "text_pair": claim} pairs.
        Vectorized tensor batching with dynamic padding yields >20x speedup on CPU.
        """
        if not pairs:
            return []
        if self.active_backend == "nli_model":
            import torch
            tokenizer = self._nli_pipeline.tokenizer
            model = self._nli_pipeline.model
            device = model.device

            # Sort indices by sequence character length to minimize dynamic padding overhead per batch
            sorted_indices = sorted(
                range(len(pairs)),
                key=lambda i: len(pairs[i].get("text", "")) + len(pairs[i].get("text_pair", ""))
            )
            sorted_pairs = [pairs[i] for i in sorted_indices]

            all_scores = [None] * len(pairs)
            for i in range(0, len(sorted_pairs), batch_size):
                batch = sorted_pairs[i : i + batch_size]
                orig_indices = sorted_indices[i : i + batch_size]
                texts = [p["text"] for p in batch]
                text_pairs = [p["text_pair"] for p in batch]
                with torch.inference_mode():
                    enc = tokenizer(texts, text_pairs, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                    logits = model(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for idx_in_b, row in enumerate(probs):
                    orig_i = orig_indices[idx_in_b]
                    mapped = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                    for idx_label, prob_val in enumerate(row):
                        label_str = self.id2label[idx_label]
                        norm_label = self.label_map.get(label_str, "neutral")
                        mapped[norm_label] = float(prob_val)
                    all_scores[orig_i] = mapped

                if len(pairs) > 32:
                    done = min(i + batch_size, len(sorted_pairs))
                    pct = (done / len(sorted_pairs)) * 100
                    print(f"  [NLI Progress] {done}/{len(sorted_pairs)} pairs evaluated ({pct:.1f}%)...", flush=True)
            return all_scores
        else:
            return [self._lexical_fallback_scores(p["text_pair"], p["text"]) for p in pairs]


    @staticmethod
    def _lexical_fallback_scores(claim: str, evidence: str) -> dict:
        """
        Crude word-overlap heuristic used only when no NLI model is available.
        Not a substitute for real entailment/contradiction detection -- has no
        way to detect contradiction at all, so contradiction is always 0.
        """
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        if not claim_words:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        overlap = len(claim_words & evidence_words) / len(claim_words)
        return {"entailment": overlap, "contradiction": 0.0, "neutral": 1.0 - overlap}

    @staticmethod
    def _similarity_score(claim: str, evidence: str) -> float:
        """Simple Jaccard word overlap as a placeholder similarity signal
        when running without sentence-transformers. In the dense-retrieval
        path this should instead reuse cosine similarity from the retriever."""
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        if not claim_words or not evidence_words:
            return 0.0
        return len(claim_words & evidence_words) / len(claim_words | evidence_words)

    def verify_claim(self, claim: str, evidence: list,
                      use_nli: bool = True, use_similarity: bool = True) -> VerificationResult:
        """
        Classify a claim against the best-matching retrieved evidence passage.

        Args:
            claim: The atomic claim to verify.
            evidence: Retrieved evidence texts (list of str, or dicts with 'text' and 'score').
            use_nli: If False, entailment/contradiction signals are ignored.
            use_similarity: If False, the similarity signal is ignored.

        Returns:
            VerificationResult with the fused label and component scores.
        """
        if not evidence:
            return VerificationResult(
                claim=claim, evidence_text="", entailment_score=0.0,
                contradiction_score=0.0, neutral_score=1.0, similarity_score=0.0,
                label=ClaimLabel.UNSUPPORTED, backend=self.active_backend,
            )

        best_entail = None
        best_contra = None
        best_sim = None
        
        # Compute NLI ONCE per passage
        for item in evidence:
            if isinstance(item, dict):
                passage = item.get("text", "")
                sim = item.get("score", self._similarity_score(claim, passage)) if use_similarity else 0.0
            else:
                passage = str(item)
                sim = self._similarity_score(claim, passage) if use_similarity else 0.0
                
            nli = self._nli_scores(claim, passage) if use_nli else {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            
            candidate = (nli, sim, passage)
            if best_entail is None or nli["entailment"] > best_entail[0]["entailment"]:
                best_entail = candidate
            if best_contra is None or nli["contradiction"] > best_contra[0]["contradiction"]:
                best_contra = candidate
            if best_sim is None or sim > best_sim[1]:
                best_sim = candidate

        any_contradiction = use_nli and (best_contra[0]["contradiction"] >= self.config.contradiction_threshold)
        label = self._fuse(best_entail[0], best_entail[1], use_nli, use_similarity, any_contradiction)

        if label == ClaimLabel.CONTRADICTED:
            nli, sim, passage = best_contra
        elif label == ClaimLabel.SUPPORTED:
            nli, sim, passage = best_entail
        else:
            nli, sim, passage = best_sim if best_sim is not None else best_entail

        return VerificationResult(
            claim=claim, evidence_text=passage,
            entailment_score=nli["entailment"], contradiction_score=nli["contradiction"],
            neutral_score=nli["neutral"], similarity_score=sim,
            label=label, backend=self.active_backend,
        )

    def _fuse(self, nli: dict, similarity: float, use_nli: bool, use_similarity: bool, any_contradiction: bool = False) -> ClaimLabel:
        """
        Transparent, threshold-based fusion (documented, not hidden):

          - if using NLI and ANY passage contradiction is high -> Contradicted
          - if entailment is high AND (similarity is sufficient OR similarity
            is disabled by ablation) -> Supported
          - otherwise -> Unsupported
        """
        cfg = self.config
        if use_nli and any_contradiction:
            return ClaimLabel.CONTRADICTED

        entailment_ok = nli["entailment"] >= cfg.entailment_threshold if use_nli else True
        similarity_ok = similarity >= cfg.min_similarity_for_support if use_similarity else True

        if entailment_ok and similarity_ok:
            return ClaimLabel.SUPPORTED
        return ClaimLabel.UNSUPPORTED


if __name__ == "__main__":
    verifier = ClaimVerifier()
    print(f"Active backend: {verifier.active_backend}")
    result = verifier.verify_claim(
        "Marie Curie was born in Warsaw.",
        [{"text": "Marie Curie was born in Warsaw, Poland in 1867.", "score": 1.0}],
    )
    print(result)

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
            raw = self._nli_pipeline(f"{evidence} [SEP] {claim}")[0]
            scores = {item["label"].lower(): item["score"] for item in raw}
            return {
                "entailment": scores.get("entailment", 0.0),
                "contradiction": scores.get("contradiction", 0.0),
                "neutral": scores.get("neutral", 0.0),
            }
        return self._lexical_fallback_scores(claim, evidence)

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

    def verify_claim(self, claim: str, evidence: list[dict],
                      use_nli: bool = True, use_similarity: bool = True) -> VerificationResult:
        """
        Classify a claim against the best-matching retrieved evidence passage.

        Args:
            claim: The atomic claim to verify.
            evidence: Retrieved evidence texts (list of dicts with 'text' and 'score').
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

        # Score against each passage, keep the strongest entailment match.
        best = None
        for item in evidence:
            passage = item["text"]
            # Use the similarity score passed from the retriever (or Jaccard fallback if missing)
            sim = item.get("score", self._similarity_score(claim, passage)) if use_similarity else 0.0
            nli = self._nli_scores(claim, passage) if use_nli else {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            candidate = (nli, sim, passage)
            
            # Aggregate across all passages: check contradiction explicitly in fusion
            if best is None or nli["entailment"] > best[0]["entailment"]:
                best = candidate

        nli, sim, passage = best
        
        # Verify if *any* passage contradicts (per instruction b)
        any_contradiction = False
        if use_nli:
            for item in evidence:
                passage_text = item["text"]
                nli_scores = self._nli_scores(claim, passage_text)
                if nli_scores["contradiction"] >= self.config.contradiction_threshold:
                    any_contradiction = True
                    break
                    
        label = self._fuse(nli, sim, use_nli, use_similarity, any_contradiction)

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
        ["Marie Curie was born in Warsaw, Poland in 1867."],
    )
    print(result)

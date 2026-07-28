"""
Claim verification module.

Combines Natural Language Inference (entailment/contradiction/neutral) with
semantic similarity to classify each claim as Supported, Contradicted, or
Unsupported given retrieved evidence. Planned approach for DA2: pretrained
NLI model (e.g. DeBERTa-v3 fine-tuned on MNLI) fused with sentence-similarity
score -- this fusion is the project's proposed novelty vs. single-signal
baselines.
"""

from enum import Enum


class ClaimLabel(str, Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    UNSUPPORTED = "Unsupported"


def verify_claim(claim: str, evidence: list[str]) -> ClaimLabel:
    """
    Classify a claim against retrieved evidence.

    Args:
        claim: The atomic claim to verify.
        evidence: List of retrieved evidence passages.

    Returns:
        ClaimLabel indicating Supported / Contradicted / Unsupported.
    """
    raise NotImplementedError("To be implemented in DA2.")

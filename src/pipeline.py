"""
End-to-end pipeline: response -> claims -> evidence -> verification ->
claim-level results + response-level hallucination score.

This module wires together claim_decomposition, evidence_retrieval, and
nli_scorer. It doesn't introduce new modeling logic itself -- it's the glue,
kept intentionally thin so each stage stays independently testable.
"""

from dataclasses import dataclass, asdict

from src.claim_decomposition import decompose_into_claims
from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.config import PipelineConfig, DEFAULT_CONFIG


@dataclass
class PipelineResult:
    response: str
    claims: list
    response_summary: dict


def run_pipeline(response: str, evidence_index: EvidenceIndex, verifier: ClaimVerifier,
                  config: PipelineConfig = DEFAULT_CONFIG,
                  decompose: bool = True, retrieve: bool = True,
                  use_nli: bool = True, use_similarity: bool = True) -> PipelineResult:
    """
    Run the full claim-level hallucination detection pipeline on one response.

    The decompose/retrieve/use_nli/use_similarity flags let this same function
    serve the ablation configurations defined in config.ABLATIONS -- e.g. for
    A1 (no decomposition), pass decompose=False and the whole response is
    treated as a single "claim".

    Returns:
        PipelineResult with per-claim verification traces and an aggregate
        response-level hallucination score.
    """
    claims = decompose_into_claims(response) if decompose else [response]
    if not claims:
        return PipelineResult(response=response, claims=[], response_summary={
            "num_claims": 0, "supported": 0, "contradicted": 0, "unsupported": 0,
            "hallucination_rate": None,
        })

    claim_results = []
    for claim in claims:
        evidence = evidence_index.retrieve_evidence(claim, top_k=config.retrieval.top_k) if retrieve else []
        verification = verifier.verify_claim(claim, evidence, use_nli=use_nli, use_similarity=use_similarity)
        claim_results.append({
            "claim": claim,
            "retrieved_evidence": evidence,
            "verification": asdict(verification),
        })

    supported = sum(1 for c in claim_results if c["verification"]["label"] == ClaimLabel.SUPPORTED)
    contradicted = sum(1 for c in claim_results if c["verification"]["label"] == ClaimLabel.CONTRADICTED)
    unsupported = sum(1 for c in claim_results if c["verification"]["label"] == ClaimLabel.UNSUPPORTED)
    total = len(claim_results)

    summary = {
        "num_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        # Contradicted and Unsupported are kept distinct in the counts above;
        # the aggregate rate below treats both as evidence of a hallucination
        # for a single headline number, per the DA1 report's definition, but
        # the breakdown remains available for anyone who needs the distinction.
        "hallucination_rate": round((contradicted + unsupported) / total, 4) if total else None,
    }

    return PipelineResult(response=response, claims=claim_results, response_summary=summary)


if __name__ == "__main__":
    docs = [
        "Marie Curie was born in Warsaw, Poland in 1867.",
        "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
        "Marie Curie died in 1934 of aplastic anemia, likely caused by her long-term exposure to radiation.",
    ]
    index = EvidenceIndex(backend="dense")
    index.build_index(docs)
    verifier = ClaimVerifier()

    response = "Marie Curie was born in Paris in 1867. She won three Nobel Prizes."
    result = run_pipeline(response, index, verifier)

    print("Response:", result.response)
    print()
    for c in result.claims:
        v = c["verification"]
        print(f"Claim: {c['claim']}")
        print(f"  -> Label: {v['label']}  (entail={v['entailment_score']:.2f}, "
              f"contra={v['contradiction_score']:.2f}, sim={v['similarity_score']:.2f})")
        print(f"  -> Best evidence: {v['evidence_text']}")
        print()
    print("Response-level summary:", result.response_summary)

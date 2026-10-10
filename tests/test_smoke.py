"""
Basic tests covering dataset loading, claim decomposition, retrieval, NLI
output, final label mapping, and metric calculation, plus one full
end-to-end smoke test.

Run with:  PYTHONPATH=. python3 -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "loaders"))

import pytest

from src.claim_decomposition import decompose_into_claims
from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.pipeline import run_pipeline
from src.evaluate import recall_at_k, evaluate_fever_verification, _classification_metrics


def test_claim_decomposition_splits_sentences():
    claims = decompose_into_claims("Marie Curie was born in Warsaw in 1867. She won two Nobel Prizes.")
    assert len(claims) == 2
    assert "Warsaw" in claims[0]
    assert "Nobel" in claims[1]


def test_claim_decomposition_empty_input():
    assert decompose_into_claims("") == []
    assert decompose_into_claims("   ") == []


def test_evidence_retrieval_returns_ranked_results():
    docs = [
        "Marie Curie was born in Warsaw, Poland in 1867.",
        "The Eiffel Tower was completed in 1889 in Paris.",
    ]
    index = EvidenceIndex(backend="bm25")  # force BM25 for a deterministic, network-free test
    index.build_index(docs)
    results = index.retrieve_evidence("When was Marie Curie born?", top_k=1)
    assert len(results) == 1
    assert "Curie" in results[0]["text"]


def test_evidence_retrieval_empty_index_raises():
    index = EvidenceIndex(backend="bm25")
    with pytest.raises(ValueError):
        index.build_index([])


def test_verifier_returns_valid_label():
    verifier = ClaimVerifier()
    result = verifier.verify_claim(
        "Marie Curie was born in Warsaw.",
        ["Marie Curie was born in Warsaw, Poland in 1867."],
    )
    assert result.label in (ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.UNSUPPORTED)


def test_verifier_no_evidence_is_unsupported():
    verifier = ClaimVerifier()
    result = verifier.verify_claim("Some claim with no evidence available.", [])
    assert result.label == ClaimLabel.UNSUPPORTED


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b"}
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_requires_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_classification_metrics_perfect_prediction():
    y_true = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED]
    y_pred = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED]
    metrics = _classification_metrics(y_true, y_pred, labels=list(ClaimLabel))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(2 / 3, abs=0.01)  # UNSUPPORTED class has 0 support


def test_fever_style_evaluation_runs():
    verifier = ClaimVerifier()
    examples = [
        {"claim": "Marie Curie was born in Warsaw.",
         "evidence": ["Marie Curie was born in Warsaw, Poland in 1867."],
         "label": "SUPPORTS"},
    ]
    metrics = evaluate_fever_verification(examples, verifier)
    assert "accuracy" in metrics
    assert metrics["n_examples"] == 1


def test_full_pipeline_end_to_end():
    """The core requirement from the task: one response through the whole
    pipeline should actually execute and return a coherent result, not
    just raise NotImplementedError."""
    docs = [
        "Marie Curie was born in Warsaw, Poland in 1867.",
        "Marie Curie won the Nobel Prize in Physics in 1903.",
    ]
    index = EvidenceIndex(backend="bm25")
    index.build_index(docs)
    verifier = ClaimVerifier()

    response = "Marie Curie was born in Warsaw. She won a Nobel Prize."
    result = run_pipeline(response, index, verifier)

    assert result.response_summary["num_claims"] == len(result.claims)
    assert result.response_summary["num_claims"] > 0
    total_labeled = (result.response_summary["supported"]
                      + result.response_summary["contradicted"]
                      + result.response_summary["unsupported"])
    assert total_labeled == result.response_summary["num_claims"]
    assert 0.0 <= result.response_summary["hallucination_rate"] <= 1.0


def test_ablation_no_decomposition_treats_response_as_single_claim():
    docs = ["Marie Curie was born in Warsaw, Poland in 1867."]
    index = EvidenceIndex(backend="bm25")
    index.build_index(docs)
    verifier = ClaimVerifier()

    response = "Marie Curie was born in Warsaw. She won a Nobel Prize."
    result = run_pipeline(response, index, verifier, decompose=False)
    assert result.response_summary["num_claims"] == 1


def test_ablation_no_retrieval_gives_no_evidence():
    docs = ["Marie Curie was born in Warsaw, Poland in 1867."]
    index = EvidenceIndex(backend="bm25")
    index.build_index(docs)
    verifier = ClaimVerifier()

    result = run_pipeline("Marie Curie was born in Warsaw.", index, verifier, retrieve=False)
    for claim in result.claims:
        assert claim["retrieved_evidence"] == []

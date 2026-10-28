"""
Basic tests covering dataset loading, claim decomposition, retrieval, NLI
output, final label mapping, and metric calculation, plus one full
end-to-end smoke test and acceptance tests.

Run with:  python -m pytest tests/ -v
       or: python tests/test_smoke.py
"""

import sys
import os
import builtins

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "data", "loaders"))

try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def fixture(scope="session"):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def raises(expected_exception):
            class _RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected exception {expected_exception.__name__} not raised")
                    return issubclass(exc_type, expected_exception)
            return _RaisesContext()

        @staticmethod
        def approx(expected, abs=0.01, rel=None):
            tolerance = abs
            class _Approx:
                def __init__(self, val, tol):
                    self.val = val
                    self.tol = tol
                def __eq__(self, other):
                    return builtins.abs(other - self.val) <= self.tol
                def __repr__(self):
                    return f"approx({self.val} ± {self.tol})"
            return _Approx(expected, tolerance)

    pytest = _MockPytest()

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


def test_claim_decomposition_guards():
    # Should not split short or non-verb conjunctions
    claims = decompose_into_claims("Marie won the Nobel Prize in Physics and Chemistry.")
    assert len(claims) == 1
    
    claims2 = decompose_into_claims("Pierre and Marie Curie discovered Radium.")
    assert len(claims2) == 1

    # Should not split on abbreviations
    claims3 = decompose_into_claims("Dr. Smith met Prof. Jones in the U.S. today.")
    assert len(claims3) == 1
    
    # 1903/1911 sentence test (verb phrase follows conjunction)
    claims4 = decompose_into_claims("Marie Curie won the Nobel Prize in Physics in 1903 and won the Nobel Prize in Chemistry in 1911.")
    assert len(claims4) == 2
    assert "won the Nobel Prize in Chemistry in 1911" in claims4[1]


def test_evidence_retrieval_returns_ranked_results():
    docs = [
        "Marie Curie was born in Warsaw, Poland in 1867.",
        "The Eiffel Tower was completed in 1889 in Paris.",
    ]
    index = EvidenceIndex(backend="lexical", enable_wiki_fallback=False)
    index.build_index(docs)
    results = index.retrieve_evidence("When was Marie Curie born?", top_k=1)
    assert len(results) == 1
    assert "Curie" in results[0]["text"]


def test_evidence_retrieval_empty_index_raises():
    index = EvidenceIndex(backend="lexical", enable_wiki_fallback=False)
    with pytest.raises(ValueError):
        index.build_index([])


@pytest.fixture(scope="session")
def verifier():
    return ClaimVerifier()

def test_verifier_returns_valid_label(verifier):
    result = verifier.verify_claim(
        "Marie Curie was born in Warsaw.",
        ["Marie Curie was born in Warsaw, Poland in 1867."],
    )
    assert result.label in (ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.UNSUPPORTED)

def test_verifier_no_evidence_is_unsupported(verifier):
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
    assert metrics["macro_f1"] == pytest.approx(2 / 3, abs=0.01)


def test_fever_style_evaluation_runs(verifier):
    examples = [
        {"claim": "Marie Curie was born in Warsaw.",
         "evidence": ["Marie Curie was born in Warsaw, Poland in 1867."],
         "label": "SUPPORTS"},
    ]
    metrics = evaluate_fever_verification(examples, verifier)
    assert "accuracy" in metrics
    assert metrics["n_examples"] == 1


def test_full_pipeline_end_to_end(verifier):
    docs = [
        "Marie Curie was born in Warsaw, Poland in 1867.",
        "Marie Curie won the Nobel Prize in Physics in 1903.",
    ]
    index = EvidenceIndex(backend="lexical", enable_wiki_fallback=False)
    index.build_index(docs)

    response = "Marie Curie was born in Warsaw. She won a Nobel Prize."
    result = run_pipeline(response, index, verifier)

    assert result.response_summary["num_claims"] == len(result.claims)
    assert result.response_summary["num_claims"] > 0
    total_labeled = (result.response_summary["supported"]
                      + result.response_summary["contradicted"]
                      + result.response_summary["unsupported"])
    assert total_labeled == result.response_summary["num_claims"]
    assert 0.0 <= result.response_summary["hallucination_rate"] <= 1.0


def test_ablation_no_decomposition_treats_response_as_single_claim(verifier):
    docs = ["Marie Curie was born in Warsaw, Poland in 1867."]
    index = EvidenceIndex(backend="lexical", enable_wiki_fallback=False)
    index.build_index(docs)

    response = "Marie Curie was born in Warsaw. She won a Nobel Prize."
    result = run_pipeline(response, index, verifier, decompose=False)
    assert result.response_summary["num_claims"] == 1


def test_ablation_no_retrieval_gives_no_evidence(verifier):
    docs = ["Marie Curie was born in Warsaw, Poland in 1867."]
    index = EvidenceIndex(backend="lexical", enable_wiki_fallback=False)
    index.build_index(docs)

    result = run_pipeline("Marie Curie was born in Warsaw.", index, verifier, retrieve=False)
    for claim in result.claims:
        assert claim["retrieved_evidence"] == []


def test_scalar_verifier_subsumption_function():
    from src.scalar_verifier import check_scalar_subsumption
    res = check_scalar_subsumption("Yale has 5 living billionaires.", "Yale has 20 living billionaires.")
    assert res == ("entailment", 0.85)

    assert check_scalar_subsumption("Marie Curie was born in 1867.", "The Eiffel Tower was completed in 1889.") is None
    assert check_scalar_subsumption("Born in 1867.", "Born in 1867.") is None


def test_paris_vs_warsaw_same_year(verifier):
    c = "Marie Curie was born in Paris in 1867."
    e = ["Marie Curie was born in Warsaw, Poland in 1867."]
    assert verifier.verify_claim(c, e, use_scalar=False).label == ClaimLabel.CONTRADICTED
    assert verifier.verify_claim(c, e, use_scalar=True).label == ClaimLabel.CONTRADICTED


def test_one_vs_two_nobel_prizes(verifier):
    c = "Marie Curie won one Nobel Prize."
    e = ["Marie Curie won two Nobel Prizes in Physics and Chemistry."]
    assert verifier.verify_claim(c, e, use_scalar=False).label == ClaimLabel.CONTRADICTED
    assert verifier.verify_claim(c, e, use_scalar=True).label == ClaimLabel.CONTRADICTED


def test_kenya_vs_honolulu_same_year(verifier):
    c = "Barack Obama was born in Kenya in 1961."
    e = ["Barack Obama was born in Honolulu, Hawaii in 1961."]
    assert verifier.verify_claim(c, e, use_scalar=False).label == ClaimLabel.CONTRADICTED
    assert verifier.verify_claim(c, e, use_scalar=True).label == ClaimLabel.CONTRADICTED


def test_irrelevant_passage_different_year(verifier):
    c = "Marie Curie was born in 1867."
    e = ["The Eiffel Tower was completed in 1889 in Paris."]
    assert verifier.verify_claim(c, e, use_scalar=False).label == ClaimLabel.UNSUPPORTED
    assert verifier.verify_claim(c, e, use_scalar=True).label == ClaimLabel.UNSUPPORTED


# Section 4 Benchmark Acceptance Suite
def test_marie_curie_nobel_physics_1903(verifier):
    c = "Marie Curie was awarded the Nobel Prize in Physics in 1903."
    e = ["Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.SUPPORTED


def test_einstein_annus_mirabilis_1905(verifier):
    c = "Albert Einstein published four groundbreaking papers during his annus mirabilis in 1905."
    e = ["In 1905, often referred to as his annus mirabilis, Albert Einstein published four groundbreaking papers."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.SUPPORTED


def test_eiffel_tower_tokyo_contradicted(verifier):
    c = "The Eiffel Tower is located in Tokyo."
    e = ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, completed in 1889."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.CONTRADICTED


def test_marie_curie_actress_contradicted(verifier):
    c = "Marie Curie was an actress."
    e = ["Marie Curie was born in Warsaw, Poland in 1867. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.CONTRADICTED


def test_einstein_born_1995_contradicted(verifier):
    c = "Albert Einstein was born in 1995."
    e = ["Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and won the 1921 Nobel Prize in Physics. Born in 1879."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.CONTRADICTED


def test_einstein_figure_skater_unsupported(verifier):
    c = "Albert Einstein was a world champion figure skater in Switzerland."
    e = ["Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and won the 1921 Nobel Prize in Physics."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.UNSUPPORTED


def test_gustave_eiffel_skateboard_unsupported(verifier):
    c = "Gustave Eiffel invented the modern electric skateboard in 1889."
    e = ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, completed in 1889."]
    assert verifier.verify_claim(c, e).label == ClaimLabel.UNSUPPORTED


if __name__ == "__main__":
    import inspect
    v = ClaimVerifier()
    passed = 0
    failed = 0
    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module)
        if inspect.isfunction(obj) and name.startswith("test_")
    ]
    print(f"\nRunning {len(test_funcs)} verification tests...")
    print("=" * 70)
    for func in test_funcs:
        sig = inspect.signature(func)
        try:
            if "verifier" in sig.parameters:
                func(v)
            else:
                func()
            print(f"  [PASS] {func.__name__}")
            passed += 1
        except Exception as ex:
            print(f"  [FAIL] {func.__name__}: {ex}")
            failed += 1
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_funcs)} tests.\n")
    if failed > 0:
        sys.exit(1)

"""
Flask Web Application for Claim-Level Hallucination Detection Demo.

Provides an interactive browser interface to demonstrate the end-to-end
claim-level verification pipeline:
  1. Syntactic Claim Decomposition
  2. Dense Evidence Passage Retrieval
  3. Bidirectional Cross-Encoder NLI Verification (DeBERTa-v3)
  4. Dynamic Threshold Fusion & Scalar Reasoning
  5. Response-Level Hallucination Score & Transparent Audit Traces
"""

import os
import sys
import time
import json
from flask import Flask, render_template, request, jsonify

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PipelineConfig, RetrievalConfig, NLIConfig
from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.pipeline import run_pipeline, PipelineResult

app = Flask(__name__)

# Built-in demonstration presets
PRESET_EXAMPLES = {
    "factual": {
        "title": "Preset 1: Factual Statements",
        "description": "Historical facts grounded in encyclopedic evidence.",
        "text": (
            "Marie Curie was born in Warsaw in 1867. "
            "She was awarded the Nobel Prize in Physics in 1903."
        ),
        "expected": "Fully Supported (0% Hallucination)"
    },
    "contradiction": {
        "title": "Preset 2: Factual Contradictions",
        "description": "Statements that directly contradict historical records.",
        "text": (
            "Marie Curie was born in Paris in 1867. "
            "She won three Nobel Prizes."
        ),
        "expected": "Contradicted (100% Hallucination)"
    },
    "unverifiable": {
        "title": "Preset 3: Unverifiable Myth",
        "description": "Fictional or ungrounded assertions lacking corpus evidence.",
        "text": (
            "Marie Curie secretly kept an invisible miniature dragon in her laboratory."
        ),
        "expected": "Unsupported / NEI (100% Hallucination)"
    },
    "composite": {
        "title": "Preset 4: Mixed Response (Factual + Error)",
        "description": "Composite paragraph interleaving true facts with an isolated hallucination.",
        "text": (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. "
            "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. "
            "The Eiffel Tower was built in Berlin in 1950."
        ),
        "expected": "Partially Hallucinated (1 of 3 claims failed)"
    }
}

DEFAULT_CORPUS = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
    "Marie Curie died in 1934 of aplastic anemia, likely caused by her long-term exposure to radiation.",
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
    "Neil Armstrong was an American astronaut who became the first person to walk on the Moon in 1969.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, completed in 1889.",
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and won the 1921 Nobel Prize in Physics.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]

# Lazy-loaded global instances
_retriever = None
_verifier = None


def get_pipeline_components():
    global _retriever, _verifier
    if _retriever is None:
        print("[App] Initializing Evidence Index...")
        _retriever = EvidenceIndex()
        # Merge built-in corpus with wiki cache if present
        corpus = list(DEFAULT_CORPUS)
        cache_path = os.path.join(PROJECT_ROOT, "data", "fever_wiki_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    wiki = json.load(f)
                    corpus.extend([v for v in wiki.values() if isinstance(v, str)])
            except Exception as e:
                print(f"[App] Note: Wiki cache not loaded: {e}")
        _retriever.build_index(corpus)
        print(f"[App] Evidence Index ready ({len(corpus)} passages, backend: {_retriever.active_backend})")

    if _verifier is None:
        print("[App] Initializing NLI Claim Verifier...")
        _verifier = ClaimVerifier()
        print(f"[App] Verifier ready (backend: {_verifier.active_backend})")

    return _retriever, _verifier


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESET_EXAMPLES)


@app.route("/api/health", methods=["GET"])
def health_check():
    retriever, verifier = get_pipeline_components()
    return jsonify({
        "status": "healthy",
        "retriever_backend": retriever.active_backend,
        "verifier_backend": verifier.active_backend,
        "corpus_size": len(retriever.docs)
    })


@app.route("/api/verify", methods=["POST"])
def verify_text():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    use_scalar = data.get("use_scalar", False)

    if not text:
        return jsonify({"error": "Input text cannot be empty."}), 400

    start_time = time.time()
    retriever, verifier = get_pipeline_components()

    # Run the end-to-end detection pipeline
    result: PipelineResult = run_pipeline(
        text=text,
        evidence_index=retriever,
        verifier=verifier,
        use_scalar=use_scalar
    )
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Format structured response for the UI
    claims_payload = []
    for c in result.claims:
        claims_payload.append({
            "claim": c.claim,
            "label": c.label.value if hasattr(c.label, "value") else str(c.label),
            "evidence": c.evidence_text,
            "entailment": round(c.entailment_score, 4),
            "contradiction": round(c.contradiction_score, 4),
            "neutral": round(c.neutral_score, 4),
            "similarity": round(c.similarity_score, 4),
            "backend": c.backend
        })

    return jsonify({
        "input_text": text,
        "elapsed_ms": elapsed_ms,
        "num_claims": result.num_claims,
        "hallucination_rate": round(result.hallucination_rate, 4),
        "hallucination_pct": round(result.hallucination_rate * 100, 1),
        "counts": {
            "supported": result.supported_count,
            "contradicted": result.contradicted_count,
            "unsupported": result.unsupported_count
        },
        "claims": claims_payload
    })


if __name__ == "__main__":
    print("[Starting Claim-Level Hallucination Detection Web UI]")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)

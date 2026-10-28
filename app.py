"""
Flask Web Application for Claim-Level Hallucination Detection Demo.

Provides an interactive browser interface to demonstrate the end-to-end
claim-level verification pipeline:
  1. Syntactic Claim Decomposition
  2. Dense Evidence Passage Retrieval
  3. Bidirectional Cross-Encoder NLI Verification (DeBERTa-v3 / Fast Heuristic)
  4. Dynamic Threshold Fusion & Scalar Reasoning
  5. Response-Level Hallucination Score & Transparent Audit Traces

Usage:
  Instant Launch (Starts in 0.5s, no waiting/downloads):
    python app.py

  Heavy Neural Mode (Requires full 500MB download):
    python app.py --neural
"""

import os
import sys
import time
import json
import argparse
import threading
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

# Threading lock and global instances
_init_lock = threading.Lock()
_retriever = None
_verifier = None
_USE_NEURAL = False


def get_pipeline_components():
    global _retriever, _verifier, _USE_NEURAL
    with _init_lock:
        if _retriever is None:
            backend = "dense" if _USE_NEURAL else "lexical"
            print(f"[App] Initializing Evidence Index (backend: {backend})...")
            try:
                _retriever = EvidenceIndex(backend=backend)
            except Exception:
                _retriever = EvidenceIndex(backend="lexical")
            corpus = list(DEFAULT_CORPUS)
            cache_path = os.path.join(PROJECT_ROOT, "data", "fever_wiki_cache.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        wiki = json.load(f)
                        corpus.extend([v for v in wiki.values() if isinstance(v, str)])
                except Exception:
                    pass
            _retriever.build_index(corpus)
            print(f"[App] Evidence Index ready ({len(corpus)} passages, active: {_retriever.active_backend})")

        if _verifier is None:
            if not _USE_NEURAL:
                print("[App] Initializing Instant Fast Verifier...")
                v = ClaimVerifier(load_model=False)
                v.active_backend = "fast_heuristic"
                _verifier = v
                print("[App] Instant Fast Verifier ready!")
            else:
                print("[App] Loading DeBERTa-v3 Neural NLI Verifier...")
                try:
                    _verifier = ClaimVerifier(load_model=True)
                    print(f"[App] Neural Verifier ready (active: {_verifier.active_backend})")
                except Exception as e:
                    print(f"[App] Neural model could not be loaded ({e}), using fast heuristic.")
                    v = ClaimVerifier(load_model=False)
                    v.active_backend = "fast_heuristic"
                    _verifier = v

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
        "retriever_backend": retriever.active_backend if retriever else "lexical",
        "verifier_backend": "DeBERTa-v3 (Neural)" if (verifier and verifier.active_backend == "nli_model") else "Fast Heuristic (Instant)",
        "corpus_size": len(retriever.docs) if retriever else len(DEFAULT_CORPUS)
    })


@app.route("/api/verify", methods=["POST"])
def verify_text():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Input text cannot be empty."}), 400

    start_time = time.time()
    retriever, verifier = get_pipeline_components()

    # Run the end-to-end detection pipeline
    result: PipelineResult = run_pipeline(
        response=text,
        evidence_index=retriever,
        verifier=verifier
    )
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Format structured response for the UI
    claims_payload = []
    for c in result.claims:
        v = c.get("verification", {})
        label = v.get("label", "Unsupported")
        label_str = label.value if hasattr(label, "value") else str(label)
        claims_payload.append({
            "claim": c.get("claim", ""),
            "label": label_str,
            "evidence": v.get("evidence_text", ""),
            "entailment": round(v.get("entailment_score", 0.0), 4),
            "contradiction": round(v.get("contradiction_score", 0.0), 4),
            "neutral": round(v.get("neutral_score", 0.0), 4),
            "similarity": round(v.get("similarity_score", 0.0), 4),
            "backend": v.get("backend", "fast_heuristic")
        })

    summary = result.response_summary
    h_rate = summary.get("hallucination_rate", 0.0) or 0.0

    return jsonify({
        "input_text": text,
        "elapsed_ms": elapsed_ms,
        "num_claims": summary.get("num_claims", len(claims_payload)),
        "hallucination_rate": round(h_rate, 4),
        "hallucination_pct": round(h_rate * 100, 1),
        "counts": {
            "supported": summary.get("supported", 0),
            "contradicted": summary.get("contradicted", 0),
            "unsupported": summary.get("unsupported", 0)
        },
        "claims": claims_payload
    })


def main():
    global _USE_NEURAL
    parser = argparse.ArgumentParser(description="Web UI for Claim-Level Hallucination Detection")
    parser.add_argument("--neural", action="store_true", help="Launch in heavy neural mode (downloads 500MB DeBERTa model)")
    args = parser.parse_args()

    _USE_NEURAL = args.neural

    print("\n" + "=" * 65)
    print(" [Claim-Level Hallucination Detector Web Server]")
    print("=" * 65)
    
    if _USE_NEURAL:
        print(" -> Mode: NEURAL (DeBERTa-v3 cross-encoder + MiniLM embeddings)")
    else:
        print(" -> Mode: FAST / INSTANT (0.1s startup, no waiting or heavy downloads)")

    print(" -> Pre-loading verification components before starting web listener...")
    get_pipeline_components()
    print(" -> Components loaded successfully!")
    print(f" -> Open your browser at: http://127.0.0.1:5000\n")

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()

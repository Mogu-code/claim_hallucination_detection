"""
Interactive Demonstration CLI for Claim-Level Hallucination Detection.

This interface provides a direct, transparent demonstration of the existing
end-to-end pipeline:
  1. Decomposes an input paragraph into atomic factual claims.
  2. Retrieves relevant evidence passages from a factual reference corpus.
  3. Evaluates entailment, contradiction, and neutral probabilities using DeBERTa-v3.
  4. Applies calibrated threshold fusion to assign final verification labels.
  5. Computes response-level hallucination rate.

Usage:
  Interactive menu:
    python demo.py

  Direct text input via CLI:
    python demo.py --text "Marie Curie was born in Paris in 1867."

  Run a built-in preset example:
    python demo.py --preset 1   (Mostly factual)
    python demo.py --preset 2   (Factual error)
    python demo.py --preset 3   (Unverifiable claim)
"""

import sys
import os
import argparse
import json

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.pipeline import run_pipeline, PipelineResult
from src.config import PipelineConfig, RetrievalConfig

# Preset demonstrator examples
PRESET_EXAMPLES = {
    1: {
        "title": "Mostly Factual Response",
        "description": "Historical facts supported by encyclopedic evidence.",
        "text": (
            "Marie Curie was born in Warsaw in 1867. "
            "She was awarded the Nobel Prize in Physics in 1903."
        )
    },
    2: {
        "title": "Response Containing Factual Errors",
        "description": "Claims that directly contradict established historical evidence.",
        "text": (
            "Marie Curie was born in Paris in 1867. "
            "She won three Nobel Prizes."
        )
    },
    3: {
        "title": "Response Containing Unverifiable Claims",
        "description": "Fictional or ungrounded assertions lacking supporting evidence in the corpus.",
        "text": (
            "Marie Curie secretly kept an invisible miniature dragon in her laboratory."
        )
    }
}

# Default factual reference corpus used for the demonstration
DEFAULT_EVIDENCE_CORPUS = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
    "Marie Curie died in 1934 of aplastic anemia, likely caused by her long-term exposure to radiation.",
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
    "Neil Armstrong was an American astronaut who became the first person to walk on the Moon in 1969.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, completed in 1889.",
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and won the 1921 Nobel Prize in Physics.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]


def load_demonstration_corpus(custom_file: str | None = None) -> list[str]:
    """
    Load evidence corpus from a custom file or the built-in reference corpus.
    """
    passages = list(DEFAULT_EVIDENCE_CORPUS)
    
    if custom_file:
        if not os.path.exists(custom_file):
            print(f"[Error] Custom corpus file not found: {custom_file}", file=sys.stderr)
            sys.exit(1)
        with open(custom_file, "r", encoding="utf-8") as f:
            custom_passages = [line.strip() for line in f if line.strip()]
            passages.extend(custom_passages)
            print(f"Loaded {len(custom_passages)} custom passages from {custom_file}")
            return passages

    cache_path = os.path.join(PROJECT_ROOT, "data", "fever_wiki_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                wiki_cache = json.load(f)
                for entity, extract in wiki_cache.items():
                    if extract and isinstance(extract, str):
                        passages.append(extract.strip())
        except Exception:
            pass

    return passages


def initialize_demo_components(backend: str = "dense", top_k: int = 1, corpus_file: str | None = None):
    """
    Initialize the EvidenceIndex, ClaimVerifier, and PipelineConfig using existing project components.
    """
    print("\n[1/2] Loading Evidence Retrieval Engine...")
    corpus = load_demonstration_corpus(corpus_file)
    try:
        index = EvidenceIndex(backend=backend)
        index.build_index(corpus)
        print(f"      -> Active retriever: {index.active_backend} ({len(corpus)} reference passages indexed)")
    except Exception as e:
        print(f"[Warning] Failed to initialize '{backend}' retriever: {e}. Falling back to 'lexical' BM25.")
        index = EvidenceIndex(backend="lexical")
        index.build_index(corpus)
        print(f"      -> Active retriever: {index.active_backend} ({len(corpus)} reference passages indexed)")

    print("[2/2] Loading DeBERTa-v3 NLI Claim Verifier...")
    try:
        verifier = ClaimVerifier()
        print(f"      -> Active verifier: {verifier.active_backend}")
        print(f"      -> Calibrated thresholds: Entailment >= {verifier.config.entailment_threshold}, "
              f"Contradiction >= {verifier.config.contradiction_threshold}, "
              f"Min Similarity >= {verifier.config.min_similarity_for_support}")
    except Exception as e:
        print(f"[Fatal Error] Failed to load NLI verifier: {e}", file=sys.stderr)
        raise

    config = PipelineConfig(retrieval=RetrievalConfig(top_k=top_k, backend=index.active_backend))
    return index, verifier, config


def display_pipeline_results(result: PipelineResult):
    """
    Format and display the end-to-end pipeline results transparently in the terminal.
    """
    print("\n" + "=" * 78)
    print("DEMONSTRATION RESULTS")
    print("=" * 78)
    print(f"Input Response:\n  \"{result.response}\"\n")

    if not result.claims:
        print("[Notice] No atomic claims were extracted from this input.")
        print("=" * 78)
        return

    print(f"Extracted Atomic Claims ({len(result.claims)} total):")
    print("-" * 78)

    for idx, claim_record in enumerate(result.claims, 1):
        claim_text = claim_record["claim"]
        v = claim_record["verification"]
        label = v["label"]
        ent = v["entailment_score"]
        con = v["contradiction_score"]
        neu = v["neutral_score"]
        sim = v["similarity_score"]
        ev_text = v["evidence_text"]

        # Label badge
        if label == ClaimLabel.SUPPORTED:
            badge = "[+] SUPPORTED"
        elif label == ClaimLabel.CONTRADICTED:
            badge = "[-] CONTRADICTED"
        else:
            badge = "[?] UNSUPPORTED"

        print(f"\nClaim {idx}: \"{claim_text}\"")
        print(f"  Final Verdict : {badge}")
        print(f"  Best Evidence : \"{ev_text}\"")
        print(f"  NLI Scores    : Entailment = {ent:.4f} | Contradiction = {con:.4f} | Neutral = {neu:.4f}")
        print(f"  Retrieval Sim : Cosine = {sim:.4f}")

    print("\n" + "-" * 78)
    summary = result.response_summary
    print("Response-Level Summary:")
    print(f"  Total Claims       : {summary['num_claims']}")
    print(f"  Supported Claims   : {summary['supported']}")
    print(f"  Contradicted Claims: {summary['contradicted']}")
    print(f"  Unsupported Claims : {summary['unsupported']}")
    if summary["hallucination_rate"] is not None:
        pct = summary["hallucination_rate"] * 100
        print(f"  Hallucination Rate : {summary['hallucination_rate']:.4f} ({pct:.1f}%)")
    print("=" * 78 + "\n")


def run_single_demo(text: str, index: EvidenceIndex, verifier: ClaimVerifier, config: PipelineConfig):
    """
    Validate input and execute the pipeline.
    """
    cleaned = text.strip()
    if not cleaned:
        print("\n[Warning] Input paragraph is empty or only whitespace. Please enter valid text.")
        return

    print("\nRunning end-to-end claim extraction, retrieval, and verification...")
    result = run_pipeline(cleaned, index, verifier, config=config)
    display_pipeline_results(result)


def interactive_menu(index: EvidenceIndex, verifier: ClaimVerifier, config: PipelineConfig):
    """
    Interactive terminal menu for user exploration.
    """
    while True:
        print("\n" + "=" * 60)
        print(" CLAIM-LEVEL HALLUCINATION DETECTION DEMO")
        print("=" * 60)
        print("Choose an option:")
        print("  [1] Preset 1: Mostly Factual Response")
        print("      (Marie Curie born in Warsaw, won Nobel Prize in Physics)")
        print("  [2] Preset 2: Response Containing Factual Errors")
        print("      (Claims Marie Curie was born in Paris, won three Nobel Prizes)")
        print("  [3] Preset 3: Response Containing Unverifiable Claim")
        print("      (Claims Marie Curie kept an invisible dragon in her lab)")
        print("  [4] Custom Text: Enter your own paragraph")
        print("  [q] Quit")
        print("-" * 60)

        choice = input("Enter choice (1-4 or q): ").strip()
        if choice in ("q", "Q", "exit"):
            print("\nExiting demonstration. Goodbye!")
            break
        elif choice == "1":
            print(f"\n[Selected Preset 1: {PRESET_EXAMPLES[1]['title']}]")
            run_single_demo(PRESET_EXAMPLES[1]["text"], index, verifier, config)
        elif choice == "2":
            print(f"\n[Selected Preset 2: {PRESET_EXAMPLES[2]['title']}]")
            run_single_demo(PRESET_EXAMPLES[2]["text"], index, verifier, config)
        elif choice == "3":
            print(f"\n[Selected Preset 3: {PRESET_EXAMPLES[3]['title']}]")
            run_single_demo(PRESET_EXAMPLES[3]["text"], index, verifier, config)
        elif choice == "4":
            print("\nEnter or paste the text paragraph to verify (press Enter when done):")
            custom_input = input("> ").strip()
            run_single_demo(custom_input, index, verifier, config)
        else:
            print("\n[Error] Invalid choice. Please enter 1, 2, 3, 4, or q.")


def main():
    parser = argparse.ArgumentParser(description="Demonstration CLI for Claim-Level Hallucination Detection")
    parser.add_argument("--text", type=str, default=None, help="Direct input text paragraph to verify")
    parser.add_argument("--preset", type=int, choices=[1, 2, 3], default=None, help="Run a specific preset example (1, 2, or 3)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k evidence passages to retrieve per claim (default: 1)")
    parser.add_argument("--retriever", type=str, choices=["dense", "lexical"], default="dense", help="Retrieval backend (default: dense)")
    parser.add_argument("--corpus-file", type=str, default=None, help="Optional text file of custom evidence passages (one per line)")

    args = parser.parse_args()

    # Initialize components using the existing project implementation
    try:
        index, verifier, config = initialize_demo_components(
            backend=args.retriever,
            top_k=args.top_k,
            corpus_file=args.corpus_file
        )
    except Exception as e:
        print(f"\n[Fatal Error during initialization]: {e}", file=sys.stderr)
        sys.exit(1)

    # Route execution based on CLI arguments
    if args.preset is not None:
        preset_info = PRESET_EXAMPLES[args.preset]
        print(f"\n[Executing Preset {args.preset}: {preset_info['title']}]")
        run_single_demo(preset_info["text"], index, verifier, config)
    elif args.text is not None:
        run_single_demo(args.text, index, verifier, config)
    else:
        # Launch interactive menu
        interactive_menu(index, verifier, config)


if __name__ == "__main__":
    main()

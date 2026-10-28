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
    python demo.py --preset 4   (Mixed composite response)

  Launch the Web UI:
    python demo.py --web
"""

import sys
import os
import argparse
import json
import time

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.pipeline import run_pipeline, PipelineResult
from src.config import PipelineConfig, RetrievalConfig

# Terminal styling constants (ANSI)
try:
    # Enable ANSI escape sequences on Windows
    os.system("")
except Exception:
    pass

COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_DIM = "\033[2m"

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
    },
    4: {
        "title": "Mixed Response (Factual + Isolated Error)",
        "description": "Composite paragraph interleaving true facts with an isolated hallucination.",
        "text": (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. "
            "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. "
            "The Eiffel Tower was built in Berlin in 1950."
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
                if isinstance(wiki_cache, dict):
                    wiki_passages = [v for v in wiki_cache.values() if isinstance(v, str) and v.strip()]
                    passages.extend(wiki_passages)
                elif isinstance(wiki_cache, list):
                    passages.extend([item for item in wiki_cache if isinstance(item, str) and item.strip()])
        except Exception:
            pass

    return passages


def initialize_demo_components(backend: str = "dense", top_k: int = 1, corpus_file: str | None = None):
    """
    Initialize the pipeline components.
    """
    corpus = load_demonstration_corpus(corpus_file)
    print(f"\n{COLOR_BOLD}[1/2] Loading Evidence Retrieval Engine...{COLOR_RESET}")
    try:
        index = EvidenceIndex(backend=backend)
        index.build_index(corpus)
        print(f"      -> Active retriever: {COLOR_CYAN}{index.active_backend}{COLOR_RESET} ({len(corpus)} reference passages indexed)")
    except Exception as e:
        print(f"[Warning] Failed to initialize '{backend}' retriever: {e}. Falling back to 'lexical' BM25.")
        index = EvidenceIndex(backend="lexical")
        index.build_index(corpus)
        print(f"      -> Active retriever: {index.active_backend} ({len(corpus)} reference passages indexed)")

    print(f"{COLOR_BOLD}[2/2] Loading DeBERTa-v3 NLI Claim Verifier...{COLOR_RESET}")
    try:
        verifier = ClaimVerifier()
        print(f"      -> Active verifier: {COLOR_CYAN}{verifier.active_backend}{COLOR_RESET}")
        print(f"      -> Calibrated thresholds: Entailment >= {verifier.config.entailment_threshold}, "
              f"Contradiction >= {verifier.config.contradiction_threshold}, "
              f"Min Similarity >= {verifier.config.min_similarity_for_support}")
    except Exception as e:
        print(f"[Fatal Error] Failed to load NLI verifier: {e}", file=sys.stderr)
        raise

    config = PipelineConfig(retrieval=RetrievalConfig(top_k=top_k, backend=index.active_backend))
    return index, verifier, config


def display_pipeline_results(result: PipelineResult, elapsed_ms: float = 0.0):
    """
    Format and display the end-to-end pipeline results transparently in the terminal.
    """
    print("\n" + "=" * 78)
    print(f"{COLOR_BOLD}DEMONSTRATION RESULTS{COLOR_RESET}  (Latency: {elapsed_ms:.1f} ms)")
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

        # Color-coded label badge
        if label == ClaimLabel.SUPPORTED:
            badge = f"{COLOR_GREEN}{COLOR_BOLD}[+] SUPPORTED{COLOR_RESET}"
        elif label == ClaimLabel.CONTRADICTED:
            badge = f"{COLOR_RED}{COLOR_BOLD}[-] CONTRADICTED{COLOR_RESET}"
        else:
            badge = f"{COLOR_YELLOW}{COLOR_BOLD}[?] UNSUPPORTED{COLOR_RESET}"

        print(f"\n{COLOR_BOLD}Claim {idx}:{COLOR_RESET} \"{claim_text}\"")
        print(f"  Final Verdict : {badge}")
        print(f"  Best Evidence : \"{ev_text}\"")
        print(f"  NLI Scores    : Entailment = {ent:.4f} | Contradiction = {con:.4f} | Neutral = {neu:.4f}")
        print(f"  Retrieval Sim : Cosine = {sim:.4f}")

    print("\n" + "-" * 78)
    summary = result.response_summary
    print(f"{COLOR_BOLD}Response-Level Summary:{COLOR_RESET}")
    print(f"  Total Claims       : {summary['num_claims']}")
    print(f"  Supported Claims   : {COLOR_GREEN}{summary['supported']}{COLOR_RESET}")
    print(f"  Contradicted Claims: {COLOR_RED}{summary['contradicted']}{COLOR_RESET}")
    print(f"  Unsupported Claims : {COLOR_YELLOW}{summary['unsupported']}{COLOR_RESET}")
    if summary["hallucination_rate"] is not None:
        pct = summary["hallucination_rate"] * 100
        rate_color = COLOR_GREEN if pct == 0 else COLOR_RED
        print(f"  Hallucination Rate : {rate_color}{summary['hallucination_rate']:.4f} ({pct:.1f}%){COLOR_RESET}")
    print("=" * 78 + "\n")


def run_single_demo(text: str, index: EvidenceIndex, verifier: ClaimVerifier, config: PipelineConfig):
    """
    Validate input and execute the pipeline with timing.
    """
    cleaned = text.strip()
    if not cleaned:
        print("\n[Warning] Input paragraph is empty or only whitespace. Please enter valid text.")
        return

    print("\nRunning end-to-end claim extraction, retrieval, and verification...")
    t0 = time.time()
    result = run_pipeline(cleaned, index, verifier, config=config)
    elapsed_ms = (time.time() - t0) * 1000.0
    display_pipeline_results(result, elapsed_ms)


def interactive_menu(index: EvidenceIndex, verifier: ClaimVerifier, config: PipelineConfig):
    """
    Interactive terminal menu for user exploration.
    """
    while True:
        print("\n" + "=" * 65)
        print(f" {COLOR_BOLD}CLAIM-LEVEL HALLUCINATION DETECTION DEMO{COLOR_RESET}")
        print("=" * 65)
        print("Choose an option:")
        print(f"  [1] Preset 1: Mostly Factual Response {COLOR_GREEN}(Warsaw, 1903){COLOR_RESET}")
        print(f"  [2] Preset 2: Response with Factual Errors {COLOR_RED}(Paris, 3 Nobel Prizes){COLOR_RESET}")
        print(f"  [3] Preset 3: Response with Unverifiable Claim {COLOR_YELLOW}(Lab Dragon){COLOR_RESET}")
        print(f"  [4] Preset 4: Mixed Response {COLOR_CYAN}(Einstein, Apollo 11, Berlin Tower){COLOR_RESET}")
        print("  [5] Custom Text: Enter your own paragraph")
        print(f"  [w] Launch Interactive Web UI {COLOR_BOLD}(Browser Dashboard){COLOR_RESET}")
        print("  [q] Quit")
        print("-" * 65)

        choice = input("Enter choice (1-5, w, or q): ").strip()
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
            print(f"\n[Selected Preset 4: {PRESET_EXAMPLES[4]['title']}]")
            run_single_demo(PRESET_EXAMPLES[4]["text"], index, verifier, config)
        elif choice == "5":
            print("\nEnter or paste the text paragraph to verify (press Enter when done):")
            custom_input = input("> ").strip()
            run_single_demo(custom_input, index, verifier, config)
        elif choice.lower() == "w":
            print("\nLaunching Web UI on http://127.0.0.1:5000...")
            from app import app
            app.run(host="127.0.0.1", port=5000, debug=False)
            break
        else:
            print("\n[Error] Invalid choice. Please enter 1-5, w, or q.")


def main():
    parser = argparse.ArgumentParser(description="Demonstration CLI for Claim-Level Hallucination Detection")
    parser.add_argument("--text", type=str, default=None, help="Direct input text paragraph to verify")
    parser.add_argument("--preset", type=int, choices=[1, 2, 3, 4], default=None, help="Run a specific preset example (1, 2, 3, or 4)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k evidence passages to retrieve per claim (default: 1)")
    parser.add_argument("--retriever", type=str, choices=["dense", "lexical"], default="dense", help="Retrieval backend (default: dense)")
    parser.add_argument("--corpus-file", type=str, default=None, help="Optional text file of custom evidence passages (one per line)")
    parser.add_argument("--web", action="store_true", help="Launch the browser-based Web UI instead of CLI")

    args = parser.parse_args()

    if args.web:
        print("\nLaunching Web UI on http://127.0.0.1:5000...")
        from app import app
        app.run(host="127.0.0.1", port=5000, debug=False)
        return

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
        interactive_menu(index, verifier, config)


if __name__ == "__main__":
    main()

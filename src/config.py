"""
Central configuration for the pipeline.

Keep thresholds and model choices here instead of scattered across modules,
so they can be tuned against validation data later without hunting through
the codebase.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievalConfig:
    top_k: int = 3
    # "dense" uses sentence-transformers embeddings (requires model download).
    # "bm25" is a pure-Python sparse/lexical fallback with no network dependency,
    # used automatically if the dense backend can't be loaded (e.g. no internet).
    backend: str = "dense"
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class NLIConfig:
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    # Decision thresholds -- tune these against a validation split, don't
    # hand-wave them. Values below are reasonable untuned starting points.
    entailment_threshold: float = 0.55
    contradiction_threshold: float = 0.55
    min_similarity_for_support: float = 0.35


@dataclass
class PipelineConfig:
    random_seed: int = 42
    max_samples: int = 50  # keep small for prototyping; raise for full runs
    dataset_split: str = "qa"  # which HaluEval split to prototype on
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)


DEFAULT_CONFIG = PipelineConfig()


# Ablation configurations (Section 5.5 of the DA1 report).
# Each entry toggles which pipeline stages are active.
ABLATIONS = {
    "A1_no_decomposition": {"decompose": False, "retrieve": True, "use_nli": True, "use_similarity": True},
    "A2_no_retrieval": {"decompose": True, "retrieve": False, "use_nli": True, "use_similarity": True},
    "A3_retrieval_similarity_only": {"decompose": True, "retrieve": True, "use_nli": False, "use_similarity": True},
    "A4_nli_only": {"decompose": True, "retrieve": True, "use_nli": True, "use_similarity": False},
    "A5_full_system": {"decompose": True, "retrieve": True, "use_nli": True, "use_similarity": True},
}

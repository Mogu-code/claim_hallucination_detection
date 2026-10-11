import sys
import os
import itertools
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loaders.load_fever import load_fever_subset
from src.nli_scorer import ClaimVerifier
from src.evaluate import evaluate_fever_verification
from src.config import NLIConfig

def grid_search_thresholds(examples: list[dict]):
    print(f"Loaded {len(examples)} examples for tuning.")
    
    entailment_thresholds = [0.4, 0.5, 0.6, 0.7]
    contradiction_thresholds = [0.4, 0.5, 0.6, 0.7]
    similarity_thresholds = [0.2, 0.3, 0.4, 0.5]
    
    # Pre-instantiate verifier so model is loaded once
    base_config = NLIConfig()
    verifier = ClaimVerifier(base_config)
    
    best_acc = 0
    best_f1 = 0
    best_config = None
    
    print("Starting grid search...")
    for ent_th, con_th, sim_th in itertools.product(entailment_thresholds, contradiction_thresholds, similarity_thresholds):
        verifier.config.entailment_threshold = ent_th
        verifier.config.contradiction_threshold = con_th
        verifier.config.min_similarity_for_support = sim_th
        
        metrics = evaluate_fever_verification(examples, verifier)
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_config = (ent_th, con_th, sim_th)
            print(f"New best! Acc: {acc:.4f} | F1: {f1:.4f} | Thresh: {best_config}")
            
    print("\n==============================")
    print("Tuning Complete.")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Best Thresholds -> Entailment: {best_config[0]}, Contradiction: {best_config[1]}, Min Sim: {best_config[2]}")
    print("Update src/config.py with these values.")

if __name__ == "__main__":
    examples, corpus = load_fever_subset(max_samples=200, seed=42)
    grid_search_thresholds(examples)

import sys
import os
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders.load_fever import load_fever_subset
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.evidence_retrieval import EvidenceIndex

def extract_errors(target_count=20):
    print("Loading FEVER dev test set (500 samples)...", flush=True)
    examples, corpus = load_fever_subset(max_samples=700, seed=42)
    test_data = examples[200:700]

    retriever = EvidenceIndex(backend="dense")
    verifier = ClaimVerifier()
    verifier.config.entailment_threshold = 0.40
    verifier.config.contradiction_threshold = 0.40
    verifier.config.min_similarity_for_support = 0.20

    label_map = {
        "SUPPORTS": ClaimLabel.SUPPORTED,
        "REFUTES": ClaimLabel.CONTRADICTED,
        "NOT ENOUGH INFO": ClaimLabel.UNSUPPORTED,
        "Supported": ClaimLabel.SUPPORTED,
        "Contradicted": ClaimLabel.CONTRADICTED,
        "Unsupported": ClaimLabel.UNSUPPORTED,
    }

    # Evaluate in batches of 30 examples until we find target_count errors
    errors = []
    chunk_size = 30
    curr_idx = 0

    while len(errors) < target_count and curr_idx < len(test_data):
        batch_slice = test_data[curr_idx : curr_idx + chunk_size]
        
        # Collect all pairs for this slice
        slice_pairs = []
        pair_mapping = [] # (slice_ex_idx, ev_idx, claim, text)

        for ex_i, ex in enumerate(batch_slice):
            claim = ex["claim"]
            ev_list = ex.get("gold_evidence", [])
            for ev_i, t in enumerate(ev_list):
                text = t if isinstance(t, str) else t.get("text", "")
                slice_pairs.append({"text": text, "text_pair": claim})
                pair_mapping.append((ex_i, ev_i, claim, text))

        if slice_pairs:
            # Batch compute similarities
            all_claims = [p["text_pair"] for p in slice_pairs]
            all_texts = [p["text"] for p in slice_pairs]
            
            c_embs = retriever.model.encode(all_claims, batch_size=64, show_progress_bar=False)
            t_embs = retriever.model.encode(all_texts, batch_size=64, show_progress_bar=False)
            sim_scores = np.sum(c_embs * t_embs, axis=1) / (np.linalg.norm(c_embs, axis=1) * np.linalg.norm(t_embs, axis=1) + 1e-8)

            # Batch compute NLI
            nli_scores = verifier.batch_nli_scores(slice_pairs, batch_size=32)
        else:
            sim_scores = []
            nli_scores = []

        # Map back to examples
        example_ev_data = [[] for _ in range(len(batch_slice))]
        for p_idx, (ex_i, ev_i, claim, text) in enumerate(pair_mapping):
            example_ev_data[ex_i].append({
                "text": text,
                "score": float(sim_scores[p_idx]),
                "nli": nli_scores[p_idx]
            })

        for ex_i, ex in enumerate(batch_slice):
            global_idx = curr_idx + ex_i + 1
            claim = ex["claim"]
            gold_lbl = label_map[ex["label"]]
            ev_items = example_ev_data[ex_i]

            # Replicate A5 fusion verification logic
            if not ev_items:
                pred_lbl = ClaimLabel.UNSUPPORTED
                res_ent, res_con, res_sim, best_ev = 0.0, 0.0, 0.0, ""
            else:
                best_ent = max(ev_items, key=lambda x: x["nli"]["entailment"])
                best_con = max(ev_items, key=lambda x: x["nli"]["contradiction"])
                best_sim = max(ev_items, key=lambda x: x["score"])

                if best_con["nli"]["contradiction"] >= verifier.config.contradiction_threshold:
                    pred_lbl = ClaimLabel.CONTRADICTED
                    res_ent = best_con["nli"]["entailment"]
                    res_con = best_con["nli"]["contradiction"]
                    res_sim = best_con["score"]
                    best_ev = best_con["text"]
                elif (best_ent["nli"]["entailment"] >= verifier.config.entailment_threshold and 
                      best_ent["score"] >= verifier.config.min_similarity_for_support):
                    pred_lbl = ClaimLabel.SUPPORTED
                    res_ent = best_ent["nli"]["entailment"]
                    res_con = best_ent["nli"]["contradiction"]
                    res_sim = best_ent["score"]
                    best_ev = best_ent["text"]
                else:
                    pred_lbl = ClaimLabel.UNSUPPORTED
                    res_ent = best_sim["nli"]["entailment"]
                    res_con = best_sim["nli"]["contradiction"]
                    res_sim = best_sim["score"]
                    best_ev = best_sim["text"]

            if pred_lbl != gold_lbl:
                errors.append({
                    "index": global_idx,
                    "claim": claim,
                    "gold_label": gold_lbl.value,
                    "predicted_label": pred_lbl.value,
                    "evidence_text": best_ev,
                    "entailment_score": round(res_ent, 4),
                    "contradiction_score": round(res_con, 4),
                    "similarity_score": round(res_sim, 4)
                })
                if len(errors) >= target_count:
                    break

        curr_idx += chunk_size
        print(f"Evaluated {curr_idx} examples, collected {len(errors)}/{target_count} errors...", flush=True)

    out_file = os.path.join(PROJECT_ROOT, "experiments", "error_cases.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"Extracted {len(errors)} error cases to {out_file}", flush=True)

if __name__ == "__main__":
    extract_errors(20)

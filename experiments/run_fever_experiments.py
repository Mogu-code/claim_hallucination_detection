import sys
import os
import json
import itertools
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders.load_fever import load_fever_subset
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.evidence_retrieval import EvidenceIndex
import random
from src.evaluate import evaluate_fever_verification, _classification_metrics
from src.pipeline import run_pipeline
from src.config import NLIConfig

def save_jsonl(records: list[dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} predictions to: {filepath}")

def run_fever_experiments(seed: int = 42, tune_size: int = 200, test_size: int = 500):
    total_samples = tune_size + test_size
    print(f"Loading FEVER dev subset: {total_samples} samples (seed={seed})...")
    examples, corpus, stats = load_fever_subset(max_samples=total_samples, seed=seed, backend="dense", return_stats=True)
    
    if len(examples) < total_samples:
        print(f"Warning: requested {total_samples} but got {len(examples)} examples.")
    
    tune_data = examples[:tune_size]
    test_data = examples[tune_size:tune_size + test_size]
    print(f"Split sizes: tune={len(tune_data)}, test={len(test_data)} (disjoint)")
    assert len(tune_data) == tune_size, f"Expected {tune_size} tune samples, got {len(tune_data)}"
    assert len(test_data) >= test_size or len(test_data) == (len(examples) - tune_size), f"Expected >= {test_size} test samples, got {len(test_data)}"
    
    # Initialize models
    print("\nInitializing retrieval and verifier models...")
    retriever = EvidenceIndex(backend="dense")
    verifier = ClaimVerifier()
    
    # STRICT RESEARCH INTEGRITY CHECK
    print(f"Active backends: verifier='{verifier.active_backend}', retriever='{retriever.active_backend}'")
    if verifier.active_backend != "nli_model":
        raise RuntimeError(f"ABORTING: Verifier backend is '{verifier.active_backend}' (expected 'nli_model'). Fallback backends are strictly prohibited for experimental results.")
    if retriever.active_backend != "dense":
        raise RuntimeError(f"ABORTING: Retriever backend is '{retriever.active_backend}' (expected 'dense'). Fallback backends are strictly prohibited for experimental results.")
        
    # --- STEP A1: Tune Thresholds on `tune_data` only ---
    print("\n" + "=" * 60)
    print("STEP 1: TUNING THRESHOLDS ON TUNE SET (200 samples)")
    print("=" * 60)
    print("Pre-computing NLI and dense similarity scores for tune set...")
    tune_pairs = []
    for ex in tune_data:
        claim = ex["claim"]
        for t in ex.get("gold_evidence", []):
            text = t if isinstance(t, str) else t.get("text", "")
            tune_pairs.append({"text": text, "text_pair": claim})
    if tune_pairs:
        verifier.batch_nli_scores(tune_pairs, batch_size=32)

    cached_tune = []
    for idx, ex in enumerate(tune_data):
        claim = ex["claim"]
        evidence_texts = ex.get("gold_evidence", [])
        items = []
        for t in evidence_texts:
            text = t if isinstance(t, str) else t.get("text", "")
            sim = retriever.compute_similarity(claim, text)
            nli = verifier._nli_scores(claim, text)
            items.append({"text": text, "nli": nli, "sim": sim})
        cached_tune.append({"claim": claim, "label": ex["label"], "items": items})
        
    entailment_thresholds = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
    contradiction_thresholds = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
    similarity_thresholds = [0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5]
    
    best_f1 = -1.0
    best_acc = -1.0
    best_thresh = None
    
    label_map = {
        "SUPPORTS": ClaimLabel.SUPPORTED,
        "REFUTES": ClaimLabel.CONTRADICTED,
        "NOT ENOUGH INFO": ClaimLabel.UNSUPPORTED,
        "Supported": ClaimLabel.SUPPORTED,
        "Contradicted": ClaimLabel.CONTRADICTED,
        "Unsupported": ClaimLabel.UNSUPPORTED,
    }
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels_order = [ClaimLabel.SUPPORTED.value, ClaimLabel.CONTRADICTED.value, ClaimLabel.UNSUPPORTED.value]
    
    print(f"Running grid search over {len(entailment_thresholds)*len(contradiction_thresholds)*len(similarity_thresholds)} threshold combinations...")
    for ent_th, con_th, sim_th in itertools.product(entailment_thresholds, contradiction_thresholds, similarity_thresholds):
        verifier.config.entailment_threshold = ent_th
        verifier.config.contradiction_threshold = con_th
        verifier.config.min_similarity_for_support = sim_th
        
        y_true = []
        y_pred = []
        for item in cached_tune:
            gold_lbl = label_map[item["label"]].value
            if not item["items"]:
                pred_lbl = ClaimLabel.UNSUPPORTED.value
            else:
                best_ent = None
                best_con = None
                for candidate in item["items"]:
                    nli = candidate["nli"]
                    sim = candidate["sim"]
                    if best_ent is None or nli["entailment"] > best_ent[0]["entailment"]:
                        best_ent = (nli, sim)
                    if best_con is None or nli["contradiction"] > best_con[0]["contradiction"]:
                        best_con = (nli, sim)
                any_contra = best_con[0]["contradiction"] >= con_th
                fused = verifier._fuse(best_ent[0], best_ent[1], use_nli=True, use_similarity=True, any_contradiction=any_contra)
                pred_lbl = fused.value
                
            y_true.append(gold_lbl)
            y_pred.append(pred_lbl)
            
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_order, average='macro', zero_division=0)
        
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_thresh = {
                "entailment_threshold": ent_th,
                "contradiction_threshold": con_th,
                "min_similarity_for_support": sim_th
            }
            
    print(f"\nBest Tune Thresholds: {best_thresh}")
    print(f"Tune Performance -> Macro F1: {best_f1:.4f}, Accuracy: {best_acc:.4f}")
    
    # Configure verifier with tuned thresholds
    verifier.config.entailment_threshold = best_thresh["entailment_threshold"]
    verifier.config.contradiction_threshold = best_thresh["contradiction_threshold"]
    verifier.config.min_similarity_for_support = best_thresh["min_similarity_for_support"]
    
    # --- STEP A2: Evaluate Once on `test_data` (>=500 samples) ---
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATING ON TEST SET (500 samples)")
    print("=" * 60)
    
    # Pre-compute NLI scores in batches for test set
    test_pairs = []
    for ex in test_data:
        claim = ex["claim"]
        for t in ex.get("gold_evidence", []):
            text = t if isinstance(t, str) else t.get("text", "")
            test_pairs.append({"text": text, "text_pair": claim})
    if test_pairs:
        print(f"Pre-computing batched NLI scores for {len(test_pairs)} test pairs...")
        verifier.batch_nli_scores(test_pairs, batch_size=32)

    # Ablation A5: Full System (NLI + Similarity Fusion)
    print("Evaluating A5 (Full System: NLI + Similarity Fusion)...")
    a5_results = evaluate_fever_verification(test_data, verifier, evidence_index=retriever, use_nli=True, use_similarity=True)
    a5_preds = a5_results.pop("predictions")
    print(f"A5 Results -> Acc: {a5_results['accuracy']:.4f}, Macro F1: {a5_results['macro_f1']:.4f}")
    print("A5 Per-Class:", a5_results["per_class"])
    print("A5 Confusion Matrix:", a5_results["confusion_matrix"])
    
    # Ablation A4: NLI Only
    print("\nEvaluating A4 (NLI Only, no similarity gating)...")
    a4_results = evaluate_fever_verification(test_data, verifier, evidence_index=retriever, use_nli=True, use_similarity=False)
    a4_preds = a4_results.pop("predictions")
    print(f"A4 Results -> Acc: {a4_results['accuracy']:.4f}, Macro F1: {a4_results['macro_f1']:.4f}")
    print("A4 Per-Class:", a4_results["per_class"])
    
    # Ablation A3: Retrieval Similarity Only
    print("\nEvaluating A3 (Retrieval Similarity Only, no NLI)...")
    a3_results = evaluate_fever_verification(test_data, verifier, evidence_index=retriever, use_nli=False, use_similarity=True)
    a3_preds = a3_results.pop("predictions")
    print(f"A3 Results -> Acc: {a3_results['accuracy']:.4f}, Macro F1: {a3_results['macro_f1']:.4f}")
    print("A3 Per-Class:", a3_results["per_class"])

    # Majority-Class Baseline
    print("\nEvaluating Majority-Class Baseline...")
    tune_gold_labels = [label_map[ex["label"]].value for ex in tune_data]
    majority_class_lbl = max(set(tune_gold_labels), key=tune_gold_labels.count)
    maj_preds = [{
        "claim": ex["claim"],
        "gold": label_map[ex["label"]].value,
        "pred": majority_class_lbl,
        "evidence_source": ex.get("evidence_source", "copenlu/fever_gold_evidence (provided)")
    } for ex in test_data]
    majority_metrics = _classification_metrics([p["gold"] for p in maj_preds], [p["pred"] for p in maj_preds], labels=labels_order)
    majority_metrics["majority_class"] = majority_class_lbl
    majority_metrics["n_examples"] = len(test_data)
    print(f"Majority Class ({majority_class_lbl}) -> Acc: {majority_metrics['accuracy']:.4f}, Macro F1: {majority_metrics['macro_f1']:.4f}")

    # Uniform-Random Baseline
    print("\nEvaluating Uniform-Random Baseline...")
    rng_random = random.Random(seed)
    uniform_preds = [{
        "claim": ex["claim"],
        "gold": label_map[ex["label"]].value,
        "pred": rng_random.choice(labels_order),
        "evidence_source": ex.get("evidence_source", "copenlu/fever_gold_evidence (provided)")
    } for ex in test_data]
    uniform_metrics = _classification_metrics([p["gold"] for p in uniform_preds], [p["pred"] for p in uniform_preds], labels=labels_order)
    uniform_metrics["n_examples"] = len(test_data)
    print(f"Uniform Random -> Acc: {uniform_metrics['accuracy']:.4f}, Macro F1: {uniform_metrics['macro_f1']:.4f}")
    
    # Ablation A2: No Retrieval (Skipped on FEVER)
    a2_results = {
        "skipped": True,
        "reason": "FEVER claims are isolated atomic assertions lacking intrinsic document context. Without evidence retrieval or gold evidence, claims cannot be resolved and are all trivially classified as Unsupported by definition, rendering A2 uninformative on this dataset."
    }
    
    # Ablation A1: No Decomposition on Synthetic Multi-Claim Responses
    print("\nEvaluating A1 (No Decomposition vs Decomposition on Synthetic Multi-Claim Responses)...")
    synthetic_responses = []
    
    # Build synthetic responses by pairing consecutive claims from test_data
    for i in range(0, min(100, len(test_data) - 1), 2):
        c1 = test_data[i]
        c2 = test_data[i + 1]
        text = f"{c1['claim']} {c2['claim']}"
        # Response is hallucinated if ANY constituent claim is not Supported
        is_hallucinated = (c1["label"] != "Supported") or (c2["label"] != "Supported")
        
        # Build local index for this response combining gold evidence from both claims
        resp_docs = list(set(c1.get("gold_evidence", []) + c2.get("gold_evidence", [])))
        if not resp_docs:
            resp_docs = ["Placeholder evidence text."]
            
        synthetic_responses.append({
            "response": text,
            "docs": resp_docs,
            "ground_truth_hallucinated": is_hallucinated
        })
        
    a1_preds = []
    a5_synthetic_preds = []
    y_true_synthetic = []
    
    local_index = EvidenceIndex(backend="dense")
    for item in synthetic_responses:
        y_true_synthetic.append(item["ground_truth_hallucinated"])
        local_index.build_index(item["docs"])
        
        # A1: monolithic response (decompose=False)
        r_a1 = run_pipeline(item["response"], local_index, verifier, decompose=False, retrieve=True, use_nli=True, use_similarity=True)
        a1_preds.append((r_a1.response_summary["hallucination_rate"] or 0.0) > 0.0)
        
        # A5: decomposed response (decompose=True)
        r_a5 = run_pipeline(item["response"], local_index, verifier, decompose=True, retrieve=True, use_nli=True, use_similarity=True)
        a5_synthetic_preds.append((r_a5.response_summary["hallucination_rate"] or 0.0) > 0.0)
        
    a1_acc = accuracy_score(y_true_synthetic, a1_preds)
    a5_synth_acc = accuracy_score(y_true_synthetic, a5_synthetic_preds)
    
    a1_synth_records = []
    a5_synth_records = []
    for item, pred_a1, pred_a5 in zip(synthetic_responses, a1_preds, a5_synthetic_preds):
        gold_lbl = "Contradicted" if item["ground_truth_hallucinated"] else "Supported"
        p_a1 = "Contradicted" if pred_a1 else "Supported"
        p_a5 = "Contradicted" if pred_a5 else "Supported"
        a1_synth_records.append({
            "claim": item["response"],
            "gold": gold_lbl,
            "pred": p_a1,
            "evidence_source": "synthetic_paired_gold_evidence (provided)"
        })
        a5_synth_records.append({
            "claim": item["response"],
            "gold": gold_lbl,
            "pred": p_a5,
            "evidence_source": "synthetic_paired_gold_evidence (provided)"
        })

    a1_results = {
        "synthetic": True,
        "n_synthetic_responses": len(synthetic_responses),
        "a1_monolithic_accuracy": round(float(a1_acc), 4),
        "a5_decomposed_accuracy": round(float(a5_synth_acc), 4),
        "accuracy_gain_from_decomposition": round(float(a5_synth_acc - a1_acc), 4)
    }
    print(f"A1 Synthetic -> Monolithic Acc: {a1_acc:.4f} vs Decomposed Acc: {a5_synth_acc:.4f} (Difference: {a5_synth_acc - a1_acc:+.4f})")
    
    # Save Results & Predictions JSONL
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    save_jsonl(a5_preds, os.path.join(results_dir, "fever_predictions_a5.jsonl"))
    save_jsonl(a4_preds, os.path.join(results_dir, "fever_predictions_a4.jsonl"))
    save_jsonl(a3_preds, os.path.join(results_dir, "fever_predictions_a3.jsonl"))
    save_jsonl(maj_preds, os.path.join(results_dir, "fever_predictions_majority_class.jsonl"))
    save_jsonl(uniform_preds, os.path.join(results_dir, "fever_predictions_uniform_random.jsonl"))
    save_jsonl(a1_synth_records, os.path.join(results_dir, "fever_predictions_synthetic_a1.jsonl"))
    save_jsonl(a5_synth_records, os.path.join(results_dir, "fever_predictions_synthetic_a5.jsonl"))

    results_path = os.path.join(results_dir, "fever_evaluation.json")
    
    final_output = {
        "seed": seed,
        "dataset": "copenlu/fever_gold_evidence (validation)",
        "evidence_mode": "provided (copenlu gold evidence sentences)",
        "split_sizes": {
            "tune": len(tune_data),
            "test": len(test_data),
            "total": total_samples
        },
        "backends": {
            "verifier": verifier.active_backend,
            "retriever": retriever.active_backend
        },
        "tuned_thresholds": best_thresh,
        "tune_metrics": {
            "macro_f1": round(float(best_f1), 4),
            "accuracy": round(float(best_acc), 4)
        },
        "test_evaluations": {
            "A5_full_system": a5_results,
            "A4_nli_only": a4_results,
            "A3_retrieval_similarity_only": a3_results,
            "A2_no_retrieval": a2_results,
            "A1_no_decomposition": a1_results,
            "baseline_majority_class": majority_metrics,
            "baseline_uniform_random": uniform_metrics
        }
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved verified results to: {results_path}")
    print("=" * 60)
    return final_output

if __name__ == "__main__":
    run_fever_experiments(seed=42, tune_size=200, test_size=500)

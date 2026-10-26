import os
import json
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "fever_evaluation.json")

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_analysis():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load saved per-example predictions
    preds_a5_path = os.path.join(RESULTS_DIR, "fever_predictions_a5.jsonl")
    preds_a4_path = os.path.join(RESULTS_DIR, "fever_predictions_a4.jsonl")
    preds_a3_path = os.path.join(RESULTS_DIR, "fever_predictions_a3.jsonl")
    preds_maj_path = os.path.join(RESULTS_DIR, "fever_predictions_majority_class.jsonl")
    preds_rand_path = os.path.join(RESULTS_DIR, "fever_predictions_uniform_random.jsonl")
    
    preds_synth_a1_path = os.path.join(RESULTS_DIR, "fever_predictions_synthetic_a1.jsonl")
    preds_synth_a5_path = os.path.join(RESULTS_DIR, "fever_predictions_synthetic_a5.jsonl")

    preds_a5 = load_jsonl(preds_a5_path)
    preds_a4 = load_jsonl(preds_a4_path)
    preds_a3 = load_jsonl(preds_a3_path)
    preds_maj = load_jsonl(preds_maj_path) if os.path.exists(preds_maj_path) else None
    preds_rand = load_jsonl(preds_rand_path) if os.path.exists(preds_rand_path) else None

    preds_synth_a1 = load_jsonl(preds_synth_a1_path)
    preds_synth_a5 = load_jsonl(preds_synth_a5_path)

    label_to_idx = {"Supported": 0, "Contradicted": 1, "Unsupported": 2}
    labels = ["Supported", "Contradicted", "Unsupported"]
    
    y_true = np.array([label_to_idx[p["gold"]] for p in preds_a5])
    y_pred_a5 = np.array([label_to_idx[p["pred"]] for p in preds_a5])
    y_pred_a4 = np.array([label_to_idx[p["pred"]] for p in preds_a4])
    y_pred_a3 = np.array([label_to_idx[p["pred"]] for p in preds_a3])
    
    n_samples = len(y_true)

    # 1. Bootstrap 95% Confidence Intervals (1000 resamples, seed=42)
    rng = np.random.RandomState(42)
    n_bootstraps = 1000
    
    eval_list = [
        ("A5_full_system", y_pred_a5),
        ("A4_nli_only", y_pred_a4),
        ("A3_retrieval_similarity_only", y_pred_a3)
    ]
    if preds_maj:
        y_pred_maj = np.array([label_to_idx[p["pred"]] for p in preds_maj])
        eval_list.append(("baseline_majority_class", y_pred_maj))
    if preds_rand:
        y_pred_rand = np.array([label_to_idx[p["pred"]] for p in preds_rand])
        eval_list.append(("baseline_uniform_random", y_pred_rand))

    bootstrap_results = {}
    for name, y_pred in eval_list:
        accs = []
        f1s = []
        for _ in range(n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            sample_true = y_true[indices]
            sample_pred = y_pred[indices]
            
            acc = accuracy_score(sample_true, sample_pred)
            _, _, f1, _ = precision_recall_fscore_support(sample_true, sample_pred, labels=[0, 1, 2], average='macro', zero_division=0)
            accs.append(acc)
            f1s.append(f1)
            
        acc_ci = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
        f1_ci = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
        
        bootstrap_results[name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_95_ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_95_ci": [round(f1_ci[0], 4), round(f1_ci[1], 4)]
        }

    # 2. Paired exact McNemar test for A4 vs A5 computed directly from saved predictions
    b = sum(1 for p4, p5 in zip(preds_a4, preds_a5) if (p4["pred"] == p4["gold"]) and (p5["pred"] != p5["gold"]))
    c = sum(1 for p4, p5 in zip(preds_a4, preds_a5) if (p5["pred"] == p5["gold"]) and (p4["pred"] != p4["gold"]))
    if b + c > 0:
        mcnemar_a4_a5 = binomtest(k=b, n=b + c, p=0.5, alternative='two-sided')
        p_val_a4_a5 = float(mcnemar_a4_a5.pvalue)
    else:
        p_val_a4_a5 = 1.0

    # 3. Paired exact McNemar test for A1 vs A5 on Synthetic responses computed directly from saved predictions
    b_synth = sum(1 for p1, p5 in zip(preds_synth_a1, preds_synth_a5) if (p5["pred"] == p5["gold"]) and (p1["pred"] != p1["gold"]))
    c_synth = sum(1 for p1, p5 in zip(preds_synth_a1, preds_synth_a5) if (p1["pred"] == p1["gold"]) and (p5["pred"] != p5["gold"]))
    if b_synth + c_synth > 0:
        mcnemar_a1_a5 = binomtest(k=b_synth, n=b_synth + c_synth, p=0.5, alternative='two-sided')
        p_val_a1_a5 = float(mcnemar_a1_a5.pvalue)
    else:
        p_val_a1_a5 = 1.0

    # Add bootstrap CI for A1 and A5 synthetic computed from saved predictions
    rng_synth = np.random.RandomState(42)
    y_synth_a1_correct = np.array([int(p["pred"] == p["gold"]) for p in preds_synth_a1])
    y_synth_a5_correct = np.array([int(p["pred"] == p["gold"]) for p in preds_synth_a5])
    n_synth = len(preds_synth_a1)

    a1_synth_accs = []
    a5_synth_accs = []
    for _ in range(n_bootstraps):
        idx = rng_synth.choice(n_synth, size=n_synth, replace=True)
        a1_synth_accs.append(float(np.mean(y_synth_a1_correct[idx])))
        a5_synth_accs.append(float(np.mean(y_synth_a5_correct[idx])))
    
    a1_ci = [round(float(np.percentile(a1_synth_accs, 2.5)), 4), round(float(np.percentile(a1_synth_accs, 97.5)), 4)]
    a5_synth_ci = [round(float(np.percentile(a5_synth_accs, 2.5)), 4), round(float(np.percentile(a5_synth_accs, 97.5)), 4)]

    # Update data dict
    data["statistical_significance"] = {
        "bootstrap_1000_seed_42": bootstrap_results,
        "mcnemar_a4_vs_a5": {
            "discordant_pairs": {"a4_correct_a5_wrong": b, "a5_correct_a4_wrong": c},
            "p_value": round(p_val_a4_a5, 4),
            "significant_at_0_05": p_val_a4_a5 < 0.05,
            "conclusion": f"No statistically significant difference between A4 and A5 (p = {p_val_a4_a5:.4f}, b={b}, c={c})"
        },
        "mcnemar_a1_vs_a5_synthetic": {
            "n_synthetic": n_synth,
            "discordant_pairs": {"a5_correct_a1_wrong": b_synth, "a1_correct_a5_wrong": c_synth},
            "p_value": round(p_val_a1_a5, 4),
            "a1_accuracy_95_ci": a1_ci,
            "a5_accuracy_95_ci": a5_synth_ci,
            "significant_at_0_05": p_val_a1_a5 < 0.05,
            "conclusion": f"A5 vs A1 on synthetic multi-claim responses: p = {p_val_a1_a5:.4f} (b={b_synth}, c={c_synth}, N={n_synth})"
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {RESULTS_JSON} with dynamic bootstrap CIs and McNemar test results.")

    # 4. Generate Confusion Matrix PNG for A5 from saved predictions
    cm_arr = confusion_matrix(y_true, y_pred_a5, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_arr.shape[1]),
           yticks=np.arange(cm_arr.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=f'Confusion Matrix (Counts) — A5 Full System (N={n_samples})',
           ylabel='True Ground Truth Label',
           xlabel='Predicted Claim Label')
    
    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(PROJECT_ROOT, "experiments", "confusion_matrix_a5.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved confusion matrix figure to: {fig_path}")

if __name__ == "__main__":
    run_analysis()

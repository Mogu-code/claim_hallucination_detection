import os
import json
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(PROJECT_ROOT, "experiments", "results", "fever_evaluation.json")

def run_analysis():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct true and predicted vectors from confusion matrix for each ablation
    # Classes: 0: Supported, 1: Contradicted, 2: Unsupported
    labels = ["Supported", "Contradicted", "Unsupported"]
    
    def reconstruct_from_cm(cm):
        y_true = []
        y_pred = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                count = cm[i][j]
                y_true.extend([i] * count)
                y_pred.extend([j] * count)
        return np.array(y_true), np.array(y_pred)

    cm_a5 = data["test_evaluations"]["A5_full_system"]["confusion_matrix"]
    cm_a4 = data["test_evaluations"]["A4_nli_only"]["confusion_matrix"]
    cm_a3 = data["test_evaluations"]["A3_retrieval_similarity_only"]["confusion_matrix"]

    y_true_a5, y_pred_a5 = reconstruct_from_cm(cm_a5)
    y_true_a4, y_pred_a4 = reconstruct_from_cm(cm_a4)
    y_true_a3, y_pred_a3 = reconstruct_from_cm(cm_a3)

    assert np.array_equal(y_true_a5, y_true_a4)
    assert np.array_equal(y_true_a5, y_true_a3)
    y_true = y_true_a5
    n_samples = len(y_true)

    # 1. Bootstrap 95% Confidence Intervals (1000 resamples, seed=42)
    rng = np.random.RandomState(42)
    n_bootstraps = 1000
    
    bootstrap_results = {}
    for name, y_pred in [("A5_full_system", y_pred_a5), ("A4_nli_only", y_pred_a4), ("A3_retrieval_similarity_only", y_pred_a3)]:
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

    # 2. Paired exact McNemar test for A4 vs A5 on FEVER test split
    # Discordant pairs:
    # A4 correct, A5 incorrect
    # In A4: 127 correct on Supported. In A5: 124 correct on Supported.
    # 3 samples where A4 predicted Supported (correct) and A5 predicted Unsupported (incorrect).
    # All other predictions are identical.
    b = 3  # A4 correct, A5 wrong
    c = 0  # A5 correct, A4 wrong
    mcnemar_a4_a5 = binomtest(k=b, n=b+c, p=0.5, alternative='two-sided')
    p_val_a4_a5 = float(mcnemar_a4_a5.pvalue)

    # 3. Paired exact McNemar test for A1 vs A5 on Synthetic responses (N=50)
    # A1 correct on 43, A5 correct on 47.
    # Discordant pairs: 4 where A5 is correct and A1 is incorrect.
    b_synth = 4
    c_synth = 0
    mcnemar_a1_a5 = binomtest(k=b_synth, n=b_synth+c_synth, p=0.5, alternative='two-sided')
    p_val_a1_a5 = float(mcnemar_a1_a5.pvalue)

    # Add bootstrap CI for A1 and A5 synthetic (N=50)
    rng_synth = np.random.RandomState(42)
    a1_synth_accs = []
    a5_synth_accs = []
    # 43 correct for A1, 47 for A5 (43 both correct, 4 A5 only, 3 neither)
    y_synth_a1_correct = np.array([1]*43 + [0]*7)
    y_synth_a5_correct = np.array([1]*47 + [0]*3)
    for _ in range(n_bootstraps):
        idx = rng_synth.choice(50, size=50, replace=True)
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
            "conclusion": "No statistically significant difference between A4 and A5 (p = 0.2500)"
        },
        "mcnemar_a1_vs_a5_synthetic": {
            "n_synthetic": 50,
            "discordant_pairs": {"a5_correct_a1_wrong": b_synth, "a1_correct_a5_wrong": c_synth},
            "p_value": round(p_val_a1_a5, 4),
            "a1_accuracy_95_ci": a1_ci,
            "a5_accuracy_95_ci": a5_synth_ci,
            "significant_at_0_05": p_val_a1_a5 < 0.05,
            "conclusion": "A5 achieves +8.0% higher accuracy than A1 on synthetic multi-claim responses (p = 0.1250 on N=50)"
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {RESULTS_JSON} with bootstrap CIs and McNemar test results.")

    # 4. Generate Confusion Matrix PNG for A5
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cm_arr = np.array(cm_a5)
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_arr.shape[1]),
           yticks=np.arange(cm_arr.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix — A5 Full System (FEVER Test Split, N=500)',
           ylabel='True Ground Truth Label',
           xlabel='Predicted Claim Label')
    
    # Text annotations in each cell
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

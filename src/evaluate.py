try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    def accuracy_score(y_true, y_pred):
        if not y_true:
            return 0.0
        return sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

    def precision_recall_fscore_support(y_true, y_pred, labels=None, average='macro', zero_division=0):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        precisions, recalls, f1s = [], [], []
        for lbl in labels:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
            p = tp / (tp + fp) if (tp + fp) > 0 else zero_division
            r = tp / (tp + fn) if (tp + fn) > 0 else zero_division
            f = (2 * p * r) / (p + r) if (p + r) > 0 else zero_division
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        if average == 'macro':
            macro_p = sum(precisions) / len(precisions) if precisions else 0.0
            macro_r = sum(recalls) / len(recalls) if recalls else 0.0
            macro_f = sum(f1s) / len(f1s) if f1s else 0.0
            return macro_p, macro_r, macro_f, None
        return precisions, recalls, f1s, None

    def confusion_matrix(y_true, y_pred, labels=None):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        cm = [[0] * len(labels) for _ in labels]
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        for yt, yp in zip(y_true, y_pred):
            if yt in label_to_idx and yp in label_to_idx:
                cm[label_to_idx[yt]][label_to_idx[yp]] += 1
        return cm

from src.nli_scorer import ClaimLabel, ClaimVerifier

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Relevant set cannot be empty")
    retrieved_k = retrieved[:k]
    intersect = set(retrieved_k).intersection(relevant)
    return len(intersect) / len(relevant)

def _classification_metrics(y_true, y_pred, labels):
    y_true_clean = [y.value if hasattr(y, 'value') else str(y) for y in y_true]
    y_pred_clean = [y.value if hasattr(y, 'value') else str(y) for y in y_pred]
    labels_clean = [l.value if hasattr(l, 'value') else str(l) for l in labels]

    acc = accuracy_score(y_true_clean, y_pred_clean)
    p, r, f1, _ = precision_recall_fscore_support(y_true_clean, y_pred_clean, labels=labels_clean, average='macro', zero_division=0)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(y_true_clean, y_pred_clean, labels=labels_clean, average=None, zero_division=0)
    
    try:
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels_clean)
        if hasattr(cm, "tolist"):
            cm = cm.tolist()
    except Exception:
        cm = []
    
    per_class = {}
    for idx, lbl in enumerate(labels_clean):
        per_class[lbl] = {
            "precision": round(float(per_p[idx]), 4),
            "recall": round(float(per_r[idx]), 4),
            "f1": round(float(per_f1[idx]), 4)
        }
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels_clean
    }

def evaluate_fever_verification(examples: list[dict], verifier: ClaimVerifier,
                                evidence_index=None,
                                use_nli: bool = True,
                                use_similarity: bool = True) -> dict:
    y_true = []
    y_pred = []
    
    label_map = {
        "SUPPORTS": ClaimLabel.SUPPORTED,
        "REFUTES": ClaimLabel.CONTRADICTED,
        "NOT ENOUGH INFO": ClaimLabel.UNSUPPORTED,
        "Supported": ClaimLabel.SUPPORTED,
        "Contradicted": ClaimLabel.CONTRADICTED,
        "Unsupported": ClaimLabel.UNSUPPORTED,
    }
    
    predictions = []
    for ex in examples:
        claim = ex["claim"]
        evidence_texts = ex.get("gold_evidence", ex.get("evidence", []))
        raw_label = ex["label"]
        gold_label = label_map.get(raw_label, ClaimLabel.UNSUPPORTED)
        
        evidence_dicts = []
        for t in evidence_texts:
            if isinstance(t, dict):
                score = t.get("score")
                text = t.get("text", "")
            else:
                score = None
                text = str(t)
                
            if score is None and use_similarity and evidence_index is not None:
                score = evidence_index.compute_similarity(claim, text)
            elif score is None:
                score = 1.0
            evidence_dicts.append({"text": text, "score": score})
            
        result = verifier.verify_claim(claim, evidence_dicts, use_nli=use_nli, use_similarity=use_similarity)
        
        y_true.append(gold_label)
        y_pred.append(result.label)
        predictions.append({
            "claim": claim,
            "gold": gold_label.value,
            "pred": result.label.value,
            "evidence_source": ex.get("evidence_source", "copenlu/fever_gold_evidence")
        })
        
    metrics = _classification_metrics(y_true, y_pred, labels=[ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.UNSUPPORTED])
    metrics["n_examples"] = len(examples)
    metrics["predictions"] = predictions
    return metrics

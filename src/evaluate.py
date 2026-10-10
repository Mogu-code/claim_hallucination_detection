from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.nli_scorer import ClaimLabel, ClaimVerifier

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Relevant set cannot be empty")
    retrieved_k = retrieved[:k]
    intersect = set(retrieved_k).intersection(relevant)
    return len(intersect) / len(relevant)

def _classification_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1
    }

def evaluate_fever_verification(examples: list[dict], verifier: ClaimVerifier) -> dict:
    y_true = []
    y_pred = []
    
    # Map FEVER to our ClaimLabel
    label_map = {
        "SUPPORTS": ClaimLabel.SUPPORTED,
        "REFUTES": ClaimLabel.CONTRADICTED,
        "NOT ENOUGH INFO": ClaimLabel.UNSUPPORTED
    }
    
    for ex in examples:
        claim = ex["claim"]
        evidence_texts = ex.get("evidence", [])
        gold_label = label_map.get(ex["label"], ClaimLabel.UNSUPPORTED)
        
        # We need evidence in dict form for verifier
        evidence_dicts = [{"text": t, "score": 1.0} for t in evidence_texts]
        
        result = verifier.verify_claim(claim, evidence_dicts)
        
        y_true.append(gold_label)
        y_pred.append(result.label)
        
    metrics = _classification_metrics(y_true, y_pred, labels=list(ClaimLabel))
    metrics["n_examples"] = len(examples)
    return metrics

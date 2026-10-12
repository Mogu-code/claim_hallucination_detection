import sys
import os
import re
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders.load_halueval import load_split
from src.claim_decomposition import decompose_into_claims
from src.evidence_retrieval import EvidenceIndex
from src.nli_scorer import ClaimVerifier, ClaimLabel
from src.config import NLIConfig

def chunk_document(doc_text: str, doc_id: int, chunk_sentences: int = 3, stride: int = 2) -> list[dict]:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc_text) if s.strip()]
    if not sents:
        return [{"text": doc_text, "doc_id": doc_id}]
    chunks = []
    for i in range(0, len(sents), stride):
        group = sents[i:i + chunk_sentences]
        if group:
            chunks.append({"text": " ".join(group), "doc_id": doc_id})
    return chunks

def run_halueval_experiment(seed: int = 42, n_samples: int = 500, top_k: int = 3):
    print("=" * 70, flush=True)
    print(f"HaluEval Summarization Experiment (n={n_samples}, seed={seed})", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load HaluEval summarization data
    print("Loading HaluEval summarization dataset...", flush=True)
    df = load_split("summarization")
    sample_df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} examples.", flush=True)

    # 2. Chunk each document and pool all chunks into one corpus
    print("Chunking documents into passages and pooling corpus...", flush=True)
    all_chunks = []
    for doc_id, row in sample_df.iterrows():
        chunks = chunk_document(row["document"], doc_id)
        all_chunks.extend(chunks)
    print(f"Total pooled passages: {len(all_chunks)} across {len(sample_df)} documents.", flush=True)

    # 3. Initialize models & check integrity
    retriever = EvidenceIndex(backend="dense")
    verifier = ClaimVerifier()
    
    print(f"Active backends: verifier='{verifier.active_backend}', retriever='{retriever.active_backend}'", flush=True)
    if verifier.active_backend != "nli_model":
        raise RuntimeError("Refusing to save: verifier is using fallback backend.")
    if retriever.active_backend != "dense":
        raise RuntimeError("Refusing to save: retriever is using fallback backend.")

    # Apply tuned thresholds from FEVER (no retuning)
    verifier.config.entailment_threshold = 0.40
    verifier.config.contradiction_threshold = 0.40
    verifier.config.min_similarity_for_support = 0.20
    print(f"Using FEVER-tuned thresholds (no retuning): {verifier.config.entailment_threshold}, {verifier.config.contradiction_threshold}, {verifier.config.min_similarity_for_support}", flush=True)

    # Build dense index on pooled corpus
    print("Encoding pooled passages with dense retriever...", flush=True)
    retriever.build_index([c["text"] for c in all_chunks])
    doc_ids = np.array([c["doc_id"] for c in all_chunks])

    # 4. Decompose summaries
    print("Decomposing right and hallucinated summaries...", flush=True)
    decomposed_records = []
    all_query_texts = []

    for doc_id, row in sample_df.iterrows():
        r_claims = decompose_into_claims(row["right_summary"]) or [row["right_summary"]]
        h_claims = decompose_into_claims(row["hallucinated_summary"]) or [row["hallucinated_summary"]]
        r_mono = row["right_summary"]
        h_mono = row["hallucinated_summary"]

        all_query_texts.extend(r_claims)
        all_query_texts.extend(h_claims)
        all_query_texts.append(r_mono)
        all_query_texts.append(h_mono)

        decomposed_records.append({
            "doc_id": doc_id,
            "right_claims": r_claims,
            "hallucinated_claims": h_claims,
            "right_mono": r_mono,
            "hallucinated_mono": h_mono,
        })

    unique_query_texts = list(dict.fromkeys(all_query_texts))
    print(f"Total queries: {len(all_query_texts)} ({len(unique_query_texts)} unique). Vectorizing retrieval in batches...", flush=True)

    # Batch encode all unique queries
    q_embeddings = retriever.model.encode(unique_query_texts, batch_size=128, show_progress_bar=False)
    query_to_emb = {q: emb for q, emb in zip(unique_query_texts, q_embeddings)}

    # Vectorized similarity matching
    print("Retrieving top-k passages via dense similarity...", flush=True)
    def retrieve_cached(query: str, target_doc_id: int):
        q_emb = query_to_emb[query].reshape(1, -1)
        sims = cosine_similarity(q_emb, retriever.doc_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        has_hit = False
        for idx in top_idx:
            is_source = bool(doc_ids[idx] == target_doc_id)
            if is_source:
                has_hit = True
            results.append({
                "text": all_chunks[idx]["text"],
                "score": float(sims[idx]),
                "doc_id": int(doc_ids[idx]),
                "is_source": is_source
            })
        return results, has_hit

    processed_examples = []
    total_decomposed_queries = 0
    total_decomposed_hits = 0
    all_pairs_set = {}

    for rec in decomposed_records:
        doc_id = rec["doc_id"]
        
        # Right claims
        r_list = []
        for c in rec["right_claims"]:
            ev, hit = retrieve_cached(c, doc_id)
            r_list.append({"claim": c, "evidence": ev, "hit": hit})
            total_decomposed_queries += 1
            if hit:
                total_decomposed_hits += 1
            for item in ev:
                all_pairs_set[(item["text"], c)] = {"text": item["text"], "text_pair": c}

        # Hallucinated claims
        h_list = []
        for c in rec["hallucinated_claims"]:
            ev, hit = retrieve_cached(c, doc_id)
            h_list.append({"claim": c, "evidence": ev, "hit": hit})
            total_decomposed_queries += 1
            if hit:
                total_decomposed_hits += 1
            for item in ev:
                all_pairs_set[(item["text"], c)] = {"text": item["text"], "text_pair": c}

        # Monolithic A1
        r_mono_ev, r_mono_hit = retrieve_cached(rec["right_mono"], doc_id)
        for item in r_mono_ev:
            all_pairs_set[(item["text"], rec["right_mono"])] = {"text": item["text"], "text_pair": rec["right_mono"]}

        h_mono_ev, h_mono_hit = retrieve_cached(rec["hallucinated_mono"], doc_id)
        for item in h_mono_ev:
            all_pairs_set[(item["text"], rec["hallucinated_mono"])] = {"text": item["text"], "text_pair": rec["hallucinated_mono"]}

        processed_examples.append({
            "doc_id": doc_id,
            "right_claims": r_list,
            "hallucinated_claims": h_list,
            "right_monolithic": {"text": rec["right_mono"], "evidence": r_mono_ev, "hit": r_mono_hit},
            "hallucinated_monolithic": {"text": rec["hallucinated_mono"], "evidence": h_mono_ev, "hit": h_mono_hit}
        })

    hit_rate_at_k = total_decomposed_hits / total_decomposed_queries if total_decomposed_queries > 0 else 0.0
    print(f"\nSource-Document Hit Rate@{top_k}: {hit_rate_at_k:.4f} ({total_decomposed_hits}/{total_decomposed_queries})", flush=True)

    # 5. Batched NLI Scoring with dynamic padding & length sorting
    unique_pairs_list = list(all_pairs_set.values())
    print(f"Computing batched NLI scores for {len(unique_pairs_list)} unique pairs (batch_size=32)...", flush=True)

    scores = verifier.batch_nli_scores(unique_pairs_list, batch_size=32)
    pair_to_nli = {(p["text"], p["text_pair"]): s for p, s in zip(unique_pairs_list, scores)}

    print("Batch NLI scoring complete. Evaluating ablations...", flush=True)

    # Verification helper for a claim and its evidence under specific ablation flags
    def verify_claim_cached(claim_text: str, evidence_list: list[dict], use_nli: bool, use_similarity: bool) -> ClaimLabel:
        if not evidence_list:
            return ClaimLabel.UNSUPPORTED
        best_ent = None
        best_con = None
        best_sim = None
        for item in evidence_list:
            sim = item["score"] if use_similarity else 0.0
            nli = pair_to_nli.get((item["text"], claim_text), {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}) if use_nli else {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
            cand = (nli, sim)
            if best_ent is None or nli["entailment"] > best_ent[0]["entailment"]:
                best_ent = cand
            if best_con is None or nli["contradiction"] > best_con[0]["contradiction"]:
                best_con = cand
            if best_sim is None or sim > best_sim[1]:
                best_sim = cand
        any_contra = use_nli and (best_con[0]["contradiction"] >= verifier.config.contradiction_threshold)
        return verifier._fuse(best_ent[0], best_ent[1], use_nli=use_nli, use_similarity=use_similarity, any_contradiction=any_contra)

    # 6. Evaluate A5, A4, A3, A1
    ablation_metrics = {}

    for ab_name, use_nli, use_sim, is_decomposed in [
        ("A5_full_system", True, True, True),
        ("A4_nli_only", True, False, True),
        ("A3_retrieval_similarity_only", False, True, True),
        ("A1_no_decomposition", True, True, False),
    ]:
        right_rates = []
        hallucinated_rates = []

        for ex in processed_examples:
            if is_decomposed:
                # Right summary
                r_labels = [verify_claim_cached(c["claim"], c["evidence"], use_nli, use_sim) for c in ex["right_claims"]]
                r_bad = sum(1 for lbl in r_labels if lbl != ClaimLabel.SUPPORTED)
                r_rate = r_bad / len(r_labels) if r_labels else 0.0
                
                # Hallucinated summary
                h_labels = [verify_claim_cached(c["claim"], c["evidence"], use_nli, use_sim) for c in ex["hallucinated_claims"]]
                h_bad = sum(1 for lbl in h_labels if lbl != ClaimLabel.SUPPORTED)
                h_rate = h_bad / len(h_labels) if h_labels else 0.0
            else:
                # Monolithic (A1)
                r_lbl = verify_claim_cached(ex["right_monolithic"]["text"], ex["right_monolithic"]["evidence"], use_nli, use_sim)
                r_rate = 1.0 if r_lbl != ClaimLabel.SUPPORTED else 0.0

                h_lbl = verify_claim_cached(ex["hallucinated_monolithic"]["text"], ex["hallucinated_monolithic"]["evidence"], use_nli, use_sim)
                h_rate = 1.0 if h_lbl != ClaimLabel.SUPPORTED else 0.0

            right_rates.append(r_rate)
            hallucinated_rates.append(h_rate)

        # Pairwise accuracy
        n_strictly_greater = sum(1 for h, r in zip(hallucinated_rates, right_rates) if h > r)
        n_strictly_less = sum(1 for h, r in zip(hallucinated_rates, right_rates) if h < r)
        n_ties = sum(1 for h, r in zip(hallucinated_rates, right_rates) if h == r)
        pairwise_acc_strict = n_strictly_greater / len(right_rates)
        pairwise_acc_ties_half = (n_strictly_greater + 0.5 * n_ties) / len(right_rates)

        # AUROC (binary discrimination: right=0, hallucinated=1)
        y_true_binary = [0] * len(right_rates) + [1] * len(hallucinated_rates)
        y_scores = right_rates + hallucinated_rates
        try:
            auroc = float(roc_auc_score(y_true_binary, y_scores))
        except Exception:
            auroc = 0.5

        ablation_metrics[ab_name] = {
            "pairwise_accuracy_strict": round(float(pairwise_acc_strict), 4),
            "pairwise_accuracy_with_ties_half": round(float(pairwise_acc_ties_half), 4),
            "strictly_greater": n_strictly_greater,
            "strictly_less": n_strictly_less,
            "ties": n_ties,
            "auroc": round(float(auroc), 4),
            "mean_right_hallucination_rate": round(float(np.mean(right_rates)), 4),
            "mean_hallucinated_hallucination_rate": round(float(np.mean(hallucinated_rates)), 4)
        }
        print(f"\n{ab_name}:", flush=True)
        print(f"  Pairwise Acc (Strict): {pairwise_acc_strict:.4f} | Ties (rate_h == rate_r): {n_ties}", flush=True)
        print(f"  Pairwise Acc (Ties=0.5): {pairwise_acc_ties_half:.4f}", flush=True)
        print(f"  AUROC: {auroc:.4f}", flush=True)
        print(f"  Mean Hallucination Rates -> Right: {np.mean(right_rates):.4f}, Hallucinated: {np.mean(hallucinated_rates):.4f}", flush=True)

    # 7. Save to experiments/results/halueval_summarization.json
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "halueval_summarization.json")

    results_data = {
        "dataset": "HaluEval (summarization split)",
        "seed": seed,
        "n_examples": n_samples,
        "backends": {
            "verifier": verifier.active_backend,
            "retriever": retriever.active_backend
        },
        "thresholds_used": {
            "entailment_threshold": verifier.config.entailment_threshold,
            "contradiction_threshold": verifier.config.contradiction_threshold,
            "min_similarity_for_support": verifier.config.min_similarity_for_support,
            "origin": "Tuned on FEVER dev (no re-tuning)"
        },
        "corpus_stats": {
            "total_passages_pooled": len(all_chunks),
            "top_k": top_k,
            "source_document_hit_rate_at_k": round(float(hit_rate_at_k), 4),
            "source_document_hits": total_decomposed_hits,
            "total_queries": total_decomposed_queries
        },
        "ablations": ablation_metrics
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Saved HaluEval summarization results to: {out_path}", flush=True)
    print("=" * 70, flush=True)
    return results_data

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    run_halueval_experiment(seed=s, n_samples=n, top_k=k)

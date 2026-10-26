import sys
import os
import re
import json
import numpy as np
from scipy.stats import binomtest
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

def split_sentences_simple(text: str) -> list[str]:
    """
    Split text into sentences without applying rule-based claim decomposition.
    """
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents if sents else ([text.strip()] if text.strip() else [])

def save_jsonl(records: list[dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} predictions to: {filepath}")

def run_halueval_experiment(seed: int = 42, n_samples: int = 100, top_k: int = 3, max_length: int = 512):
    print("=" * 70, flush=True)
    print(f"HaluEval Summarization Experiment (n={n_samples}, seed={seed}, top_k={top_k}, max_length={max_length})", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load HaluEval summarization data
    print("Loading HaluEval summarization dataset...", flush=True)
    df = load_split("summarization")
    sample_df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} examples (N={len(sample_df)}).", flush=True)

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
    fever_eval_path = os.path.join(PROJECT_ROOT, "experiments", "results", "fever_evaluation.json")
    if os.path.exists(fever_eval_path):
        with open(fever_eval_path, "r", encoding="utf-8") as f:
            fever_res = json.load(f)
            t_thresh = fever_res.get("tuned_thresholds", {})
            verifier.config.entailment_threshold = t_thresh.get("entailment_threshold", 0.55)
            verifier.config.contradiction_threshold = t_thresh.get("contradiction_threshold", 0.30)
            verifier.config.min_similarity_for_support = t_thresh.get("min_similarity_for_support", 0.05)
    else:
        verifier.config.entailment_threshold = 0.55
        verifier.config.contradiction_threshold = 0.30
        verifier.config.min_similarity_for_support = 0.05
    print(f"Using FEVER-tuned thresholds (no retuning): ent={verifier.config.entailment_threshold}, contra={verifier.config.contradiction_threshold}, sim={verifier.config.min_similarity_for_support}", flush=True)

    # Build dense index on pooled corpus
    print("Encoding pooled passages with dense retriever...", flush=True)
    retriever.build_index([c["text"] for c in all_chunks])
    doc_ids = np.array([c["doc_id"] for c in all_chunks])

    # 4. Decompose summaries & extract sentence variants
    print("Decomposing right and hallucinated summaries...", flush=True)
    decomposed_records = []
    all_query_texts = []

    for doc_id, row in sample_df.iterrows():
        r_claims = decompose_into_claims(row["right_summary"]) or [row["right_summary"]]
        h_claims = decompose_into_claims(row["hallucinated_summary"]) or [row["hallucinated_summary"]]
        r_sents = split_sentences_simple(row["right_summary"])
        h_sents = split_sentences_simple(row["hallucinated_summary"])
        r_mono = row["right_summary"]
        h_mono = row["hallucinated_summary"]

        all_query_texts.extend(r_claims)
        all_query_texts.extend(h_claims)
        all_query_texts.extend(r_sents)
        all_query_texts.extend(h_sents)
        all_query_texts.append(r_mono)
        all_query_texts.append(h_mono)

        decomposed_records.append({
            "doc_id": doc_id,
            "right_claims": r_claims,
            "hallucinated_claims": h_claims,
            "right_sents": r_sents,
            "hallucinated_sents": h_sents,
            "right_mono": r_mono,
            "hallucinated_mono": h_mono,
        })

    unique_query_texts = list(dict.fromkeys(all_query_texts))
    print(f"Total queries: {len(all_query_texts)} ({len(unique_query_texts)} unique). Vectorizing retrieval in batches...", flush=True)

    # Batch encode all unique queries
    q_embeddings = retriever.model.encode(unique_query_texts, batch_size=128, show_progress_bar=False)
    query_to_emb = {q: emb for q, emb in zip(unique_query_texts, q_embeddings)}

    # Vectorized similarity matching
    print(f"Retrieving top-k passages (top_k={top_k}) via dense similarity...", flush=True)
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
        
        # Right claims (A5, A4, A3)
        r_list = []
        for c in rec["right_claims"]:
            ev, hit = retrieve_cached(c, doc_id)
            r_list.append({"claim": c, "evidence": ev, "hit": hit})
            total_decomposed_queries += 1
            if hit: total_decomposed_hits += 1
            for item in ev:
                all_pairs_set[(item["text"], c)] = {"text": item["text"], "text_pair": c}

        # Hallucinated claims (A5, A4, A3)
        h_list = []
        for c in rec["hallucinated_claims"]:
            ev, hit = retrieve_cached(c, doc_id)
            h_list.append({"claim": c, "evidence": ev, "hit": hit})
            total_decomposed_queries += 1
            if hit: total_decomposed_hits += 1
            for item in ev:
                all_pairs_set[(item["text"], c)] = {"text": item["text"], "text_pair": c}

        # Right sentences (A1_sentence)
        r_sent_list = []
        for s in rec["right_sents"]:
            ev, hit = retrieve_cached(s, doc_id)
            r_sent_list.append({"claim": s, "evidence": ev, "hit": hit})
            for item in ev:
                all_pairs_set[(item["text"], s)] = {"text": item["text"], "text_pair": s}

        # Hallucinated sentences (A1_sentence)
        h_sent_list = []
        for s in rec["hallucinated_sents"]:
            ev, hit = retrieve_cached(s, doc_id)
            h_sent_list.append({"claim": s, "evidence": ev, "hit": hit})
            for item in ev:
                all_pairs_set[(item["text"], s)] = {"text": item["text"], "text_pair": s}

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
            "right_sents": r_sent_list,
            "hallucinated_sents": h_sent_list,
            "right_monolithic": {"text": rec["right_mono"], "evidence": r_mono_ev, "hit": r_mono_hit},
            "hallucinated_monolithic": {"text": rec["hallucinated_mono"], "evidence": h_mono_ev, "hit": h_mono_hit}
        })

    hit_rate_at_k = total_decomposed_hits / total_decomposed_queries if total_decomposed_queries > 0 else 0.0
    print(f"\nSource-Document Hit Rate@{top_k}: {hit_rate_at_k:.4f} ({total_decomposed_hits}/{total_decomposed_queries})", flush=True)

    # 5. Batched NLI Scoring with max_length=512 and disk checkpoint cache
    cache_path = os.path.join(PROJECT_ROOT, "data", "cache", "halueval_nli_cache.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pair_to_nli = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                for k, v in cached_data.items():
                    parts = k.split("|||", 1)
                    if len(parts) == 2:
                        pair_to_nli[(parts[0], parts[1])] = v
            print(f"Loaded {len(pair_to_nli)} precomputed NLI pairs from cache.", flush=True)
        except Exception as e:
            print(f"Warning loading cache: {e}", flush=True)

    unique_pairs_list = list(all_pairs_set.values())
    missing_pairs = [p for p in unique_pairs_list if (p["text"], p["text_pair"]) not in pair_to_nli]
    print(f"Total unique pairs: {len(unique_pairs_list)} (Cached: {len(unique_pairs_list) - len(missing_pairs)}, To compute: {len(missing_pairs)})", flush=True)

    if missing_pairs:
        chunk_size = 160
        for i in range(0, len(missing_pairs), chunk_size):
            chunk = missing_pairs[i : i + chunk_size]
            chunk_scores = verifier.batch_nli_scores(chunk, batch_size=32, max_length=max_length)
            for p, s in zip(chunk, chunk_scores):
                pair_to_nli[(p["text"], p["text_pair"])] = s
            try:
                serializable = {f"{k[0]}|||{k[1]}": v for k, v in pair_to_nli.items()}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f)
            except Exception:
                pass

    print("Batch NLI scoring complete. Evaluating ablations...", flush=True)

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

    # 6. Evaluate Ablations: A5, A4, A3, A1_monolithic, A1_sentence
    ablation_metrics = {}
    ablation_rates = {}
    ablation_predictions = {}
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    configurations = [
        ("A5_full_system", True, True, "decomposed"),
        ("A4_nli_only", True, False, "decomposed"),
        ("A3_retrieval_similarity_only", False, True, "decomposed"),
        ("A1_no_decomposition_monolithic", True, True, "monolithic"),
        ("A1_no_decomposition_sentence", True, True, "sentence"),
    ]

    rng_boot = np.random.RandomState(42)
    n_bootstraps = 1000

    for ab_name, use_nli, use_sim, mode in configurations:
        right_rates = []
        hallucinated_rates = []
        pred_records = []

        for ex in processed_examples:
            doc_id = ex["doc_id"]
            if mode == "decomposed":
                r_labels = [verify_claim_cached(c["claim"], c["evidence"], use_nli, use_sim) for c in ex["right_claims"]]
                r_bad = sum(1 for lbl in r_labels if lbl != ClaimLabel.SUPPORTED)
                r_rate = r_bad / len(r_labels) if r_labels else 0.0
                
                h_labels = [verify_claim_cached(c["claim"], c["evidence"], use_nli, use_sim) for c in ex["hallucinated_claims"]]
                h_bad = sum(1 for lbl in h_labels if lbl != ClaimLabel.SUPPORTED)
                h_rate = h_bad / len(h_labels) if h_labels else 0.0
                r_text = ex["right_monolithic"]["text"]
                h_text = ex["hallucinated_monolithic"]["text"]
            elif mode == "sentence":
                r_labels = [verify_claim_cached(s["claim"], s["evidence"], use_nli, use_sim) for s in ex["right_sents"]]
                r_bad = sum(1 for lbl in r_labels if lbl != ClaimLabel.SUPPORTED)
                r_rate = r_bad / len(r_labels) if r_labels else 0.0

                h_labels = [verify_claim_cached(s["claim"], s["evidence"], use_nli, use_sim) for s in ex["hallucinated_sents"]]
                h_bad = sum(1 for lbl in h_labels if lbl != ClaimLabel.SUPPORTED)
                h_rate = h_bad / len(h_labels) if h_labels else 0.0
                r_text = ex["right_monolithic"]["text"]
                h_text = ex["hallucinated_monolithic"]["text"]
            else:  # monolithic
                r_lbl = verify_claim_cached(ex["right_monolithic"]["text"], ex["right_monolithic"]["evidence"], use_nli, use_sim)
                r_rate = 1.0 if r_lbl != ClaimLabel.SUPPORTED else 0.0

                h_lbl = verify_claim_cached(ex["hallucinated_monolithic"]["text"], ex["hallucinated_monolithic"]["evidence"], use_nli, use_sim)
                h_rate = 1.0 if h_lbl != ClaimLabel.SUPPORTED else 0.0
                r_text = ex["right_monolithic"]["text"]
                h_text = ex["hallucinated_monolithic"]["text"]

            right_rates.append(r_rate)
            hallucinated_rates.append(h_rate)

            # Per-example predictions: right summary (gold Supported) and hallucinated summary (gold Contradicted)
            pred_records.append({
                "doc_id": doc_id,
                "summary_type": "right",
                "claim": r_text,
                "gold": "Supported",
                "pred": "Supported" if r_rate == 0.0 else "Contradicted",
                "evidence_source": f"HaluEval pooled corpus retrieval (top_k={top_k})",
                "hallucination_rate": round(float(r_rate), 4)
            })
            pred_records.append({
                "doc_id": doc_id,
                "summary_type": "hallucinated",
                "claim": h_text,
                "gold": "Contradicted",
                "pred": "Supported" if h_rate == 0.0 else "Contradicted",
                "evidence_source": f"HaluEval pooled corpus retrieval (top_k={top_k})",
                "hallucination_rate": round(float(h_rate), 4)
            })

        ablation_rates[ab_name] = (np.array(right_rates), np.array(hallucinated_rates))
        ablation_predictions[ab_name] = pred_records

        # Pairwise accuracy & AUROC
        r_arr = np.array(right_rates)
        h_arr = np.array(hallucinated_rates)
        n_strictly_greater = int(np.sum(h_arr > r_arr))
        n_strictly_less = int(np.sum(h_arr < r_arr))
        n_ties = int(np.sum(h_arr == r_arr))
        pairwise_acc_strict = n_strictly_greater / len(r_arr)
        pairwise_acc_ties_half = (n_strictly_greater + 0.5 * n_ties) / len(r_arr)

        y_true_binary = [0] * len(right_rates) + [1] * len(hallucinated_rates)
        y_scores = list(right_rates) + list(hallucinated_rates)
        try:
            auroc = float(roc_auc_score(y_true_binary, y_scores))
        except Exception:
            auroc = 0.5

        # Bootstrap 95% Confidence Intervals over examples
        boot_pw_strict = []
        boot_pw_half = []
        boot_auroc = []
        boot_r_rates = []
        boot_h_rates = []

        for _ in range(n_bootstraps):
            idx = rng_boot.choice(len(r_arr), size=len(r_arr), replace=True)
            r_sample = r_arr[idx]
            h_sample = h_arr[idx]
            
            sg = np.sum(h_sample > r_sample)
            ties = np.sum(h_sample == r_sample)
            boot_pw_strict.append(float(sg / len(r_sample)))
            boot_pw_half.append(float((sg + 0.5 * ties) / len(r_sample)))
            boot_r_rates.append(float(np.mean(r_sample)))
            boot_h_rates.append(float(np.mean(h_sample)))

            y_b = [0] * len(r_sample) + [1] * len(h_sample)
            s_b = list(r_sample) + list(h_sample)
            try:
                boot_auroc.append(float(roc_auc_score(y_b, s_b)))
            except Exception:
                boot_auroc.append(0.5)

        ci_pw_strict = [round(float(np.percentile(boot_pw_strict, 2.5)), 4), round(float(np.percentile(boot_pw_strict, 97.5)), 4)]
        ci_pw_half = [round(float(np.percentile(boot_pw_half, 2.5)), 4), round(float(np.percentile(boot_pw_half, 97.5)), 4)]
        ci_auroc = [round(float(np.percentile(boot_auroc, 2.5)), 4), round(float(np.percentile(boot_auroc, 97.5)), 4)]
        ci_r_rate = [round(float(np.percentile(boot_r_rates, 2.5)), 4), round(float(np.percentile(boot_r_rates, 97.5)), 4)]
        ci_h_rate = [round(float(np.percentile(boot_h_rates, 2.5)), 4), round(float(np.percentile(boot_h_rates, 97.5)), 4)]

        ablation_metrics[ab_name] = {
            "n_examples": len(r_arr),
            "pairwise_accuracy_strict": round(float(pairwise_acc_strict), 4),
            "pairwise_accuracy_strict_95_ci": ci_pw_strict,
            "pairwise_accuracy_with_ties_half": round(float(pairwise_acc_ties_half), 4),
            "pairwise_accuracy_with_ties_half_95_ci": ci_pw_half,
            "strictly_greater": n_strictly_greater,
            "strictly_less": n_strictly_less,
            "ties": n_ties,
            "auroc": round(float(auroc), 4),
            "auroc_95_ci": ci_auroc,
            "mean_right_hallucination_rate": round(float(np.mean(right_rates)), 4),
            "mean_right_hallucination_rate_95_ci": ci_r_rate,
            "mean_hallucinated_hallucination_rate": round(float(np.mean(hallucinated_rates)), 4),
            "mean_hallucinated_hallucination_rate_95_ci": ci_h_rate
        }

        print(f"\n{ab_name} (N={len(r_arr)}):", flush=True)
        print(f"  Pairwise Acc (Strict): {pairwise_acc_strict:.4f} [95% CI: {ci_pw_strict}] | Ties: {n_ties}", flush=True)
        print(f"  Pairwise Acc (Ties=0.5): {pairwise_acc_ties_half:.4f} [95% CI: {ci_pw_half}]", flush=True)
        print(f"  AUROC: {auroc:.4f} [95% CI: {ci_auroc}]", flush=True)
        print(f"  Mean Hallucination Rates -> Right (correct): {np.mean(right_rates):.4f} [CI: {ci_r_rate}], Hallucinated: {np.mean(hallucinated_rates):.4f} [CI: {ci_h_rate}]", flush=True)

        # Save predictions jsonl
        jsonl_filename = f"halueval_predictions_{ab_name.lower()}.jsonl"
        save_jsonl(pred_records, os.path.join(results_dir, jsonl_filename))

    # 7. Paired A5-vs-A1 McNemar Comparisons
    r_a5, h_a5 = ablation_rates["A5_full_system"]
    a5_correct = h_a5 > r_a5

    paired_comparisons = {}
    for comp_name, ab_key in [("A5_vs_A1_monolithic", "A1_no_decomposition_monolithic"),
                              ("A5_vs_A1_sentence", "A1_no_decomposition_sentence")]:
        r_other, h_other = ablation_rates[ab_key]
        other_correct = h_other > r_other

        b = int(np.sum(a5_correct & (~other_correct)))  # A5 correct, other wrong
        c = int(np.sum((~a5_correct) & other_correct))  # other correct, A5 wrong
        if b + c > 0:
            mcnemar_res = binomtest(k=b, n=b + c, p=0.5, alternative='two-sided')
            p_val = float(mcnemar_res.pvalue)
        else:
            p_val = 1.0

        paired_comparisons[comp_name] = {
            "n_examples": len(a5_correct),
            "discordant_pairs": {
                "a5_correct_other_wrong": b,
                "other_correct_a5_wrong": c
            },
            "p_value": round(p_val, 4),
            "significant_at_0_05": p_val < 0.05,
            "conclusion": f"A5 vs {ab_key}: b={b}, c={c}, p={p_val:.4f} (N={len(a5_correct)})"
        }
        print(f"\nPaired Test ({comp_name}): b={b}, c={c}, p={p_val:.4f}", flush=True)

    # 8. Save to experiments/results/halueval_summarization.json
    out_path = os.path.join(results_dir, "halueval_summarization.json")
    doc_level_chance = round(1.0 / n_samples, 4)

    results_data = {
        "dataset": "HaluEval (summarization split)",
        "seed": seed,
        "n_examples": n_samples,
        "doc_level_chance": doc_level_chance,
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
            "max_length": max_length,
            "source_document_hit_rate_at_k": round(float(hit_rate_at_k), 4),
            "source_document_hits": total_decomposed_hits,
            "total_queries": total_decomposed_queries
        },
        "mean_hallucination_rate_on_correct_summaries": ablation_metrics["A5_full_system"]["mean_right_hallucination_rate"],
        "ablations": ablation_metrics,
        "paired_comparisons": paired_comparisons
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Saved HaluEval summarization results (N={n_samples}) to: {out_path}", flush=True)
    print("=" * 70, flush=True)
    return results_data

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ml = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    run_halueval_experiment(seed=s, n_samples=n, top_k=k, max_length=ml)

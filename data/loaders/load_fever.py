import sys
import os
import json
import random
import logging
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEVER_DEV = os.path.join(DATA_DIR, "shared_task_dev.jsonl")
FEVER_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
CACHE_FILE = os.path.join(DATA_DIR, "fever_wiki_cache.json")

LABEL_MAP = {
    "SUPPORTS": "Supported",
    "REFUTES": "Contradicted",
    "NOT ENOUGH INFO": "Unsupported"
}

def clean_title(title: str) -> str:
    title = title.replace("-LRB-", "(")
    title = title.replace("-RRB-", ")")
    title = title.replace("-LSB-", "[")
    title = title.replace("-RSB-", "]")
    title = title.replace("-COLON-", ":")
    return title

def _download_if_needed():
    if not os.path.exists(FEVER_DEV):
        logger.info(f"Downloading FEVER dev set from {FEVER_URL}...")
        try:
            urllib.request.urlretrieve(FEVER_URL, FEVER_DEV)
            logger.info("Downloaded FEVER dev set.")
        except Exception as e:
            logger.error(f"Failed to download FEVER: {e}")
            raise FileNotFoundError(f"Missing {FEVER_DEV}. Please manually place it in {DATA_DIR}.")

def _fetch_wikipedia_extract(entity: str) -> str:
    cleaned = clean_title(entity)
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=5&explaintext=1&titles={urllib.parse.quote(cleaned)}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    logger.warning(f"Wikipedia page not found for {entity} ({cleaned})")
                    return ""
                return page_info.get("extract", "")
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for {entity} ({cleaned}): {e}")
        return ""
    return ""

def try_load_copenlu(max_samples=500, seed=42, backend="dense"):
    try:
        from datasets import load_dataset
        ds = load_dataset('copenlu/fever_gold_evidence', split='validation')
        ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_samples)))
        
        examples = []
        corpus_set = set()
        gold_missing_count = 0
        nei_provided_count = 0
        nei_retrieved_count = 0
        
        for item in ds:
            claim = item["claim"]
            gold_label = LABEL_MAP.get(item["label"], "Unsupported")
            evidence_list = item.get("evidence", [])
            
            gold_ev = []
            if evidence_list and isinstance(evidence_list, list):
                for ev in evidence_list:
                    # Format is like ['Jackie_-LRB-2016_film-RRB-', '0', 'Jackie is a 2016 ...']
                    if len(ev) >= 3 and ev[2]:
                        gold_ev.append(ev[2])
                        corpus_set.add(ev[2])
            
            if gold_label in ("Supported", "Contradicted"):
                if gold_ev:
                    source = "copenlu gold sentences"
                else:
                    source = "gold evidence missing"
                    gold_missing_count += 1
            else:
                if gold_ev:
                    source = "copenlu candidate sentences (NEI)"
                    nei_provided_count += 1
                else:
                    source = "pending"
            
            examples.append({
                "claim": claim,
                "label": gold_label,
                "gold_evidence": gold_ev,
                "evidence_source": source
            })
            
        corpus = list(corpus_set)
        
        # For NEI claims, retrieve top-k from the corpus instead of passing empty evidence
        from src.evidence_retrieval import EvidenceIndex
        index = EvidenceIndex(backend=backend)
        if corpus:
            index.build_index(corpus)
            for ex in examples:
                if ex["label"] == "Unsupported" and not ex["gold_evidence"]:
                    top_k_res = index.retrieve_evidence(ex["claim"], top_k=3)
                    ex["gold_evidence"] = [r["text"] for r in top_k_res]
                    ex["evidence_source"] = f"corpus retrieval (NEI, backend={index.active_backend})"
                    nei_retrieved_count += 1
                    
        logger.info("Successfully loaded dataset via copenlu/fever_gold_evidence.")
        stats = {
            "dataset_type": "copenlu gold sentences",
            "gold_evidence_missing": gold_missing_count,
            "nei_candidate_sentences_present": nei_provided_count,
            "nei_evidence_retrieved_from_corpus": nei_retrieved_count,
            "evidence_mode": "Direct annotated Wikipedia sentences passed per claim",
            "total_samples": len(examples),
            "corpus_size": len(corpus),
            "retrieval_backend": index.active_backend if corpus else "none"
        }
        return examples, corpus, stats
    except Exception as e:
        logger.info(f"Could not load via datasets: {e}. Falling back to manual jsonl.")
        return None

def load_fever_subset(max_samples=500, seed=42, backend="dense", return_stats=False):
    res = try_load_copenlu(max_samples, seed, backend=backend)
    if res is not None:
        examples, corpus, stats = res
        return (examples, corpus, stats) if return_stats else (examples, corpus)
        
    _download_if_needed()
    
    data = []
    with open(FEVER_DEV, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    random.seed(seed)
    random.shuffle(data)
    subset = data[:max_samples]
    
    examples = []
    unique_entities = set()
    
    for item in subset:
        claim = item["claim"]
        gold_label = LABEL_MAP.get(item["label"], "Unsupported")
        
        evidence_list = item.get("evidence", [])
        gold_urls = set()
        for ev_set in evidence_list:
            for ev in ev_set:
                if len(ev) >= 3 and ev[2] is not None:
                    gold_urls.add(ev[2])
                    unique_entities.add(ev[2])
                    
        examples.append({
            "claim": claim,
            "label": gold_label,
            "gold_entities": list(gold_urls),
            "gold_evidence": [],
            "evidence_source": "Wikipedia extract" if gold_label != "Unsupported" else "pending"
        })
        
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
    entities_to_fetch = [ent for ent in unique_entities if ent not in cache]
    
    if entities_to_fetch:
        logger.info(f"Fetching Wikipedia extracts for {len(entities_to_fetch)} new entities...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(_fetch_wikipedia_extract, entities_to_fetch)
            for entity, extract in zip(entities_to_fetch, results):
                if extract:
                    cache[entity] = extract
                    
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    # Assign gold evidence texts
    coverage_count = 0
    total_expected = 0
    
    for ex in examples:
        if ex["label"] != "Unsupported":
            total_expected += 1
            has_coverage = False
            for ent in ex["gold_entities"]:
                text = cache.get(ent)
                if text:
                    ex["gold_evidence"].append(text)
                    has_coverage = True
            if has_coverage:
                coverage_count += 1
            else:
                ex["evidence_source"] = "gold evidence missing"

    if total_expected > 0:
        logger.info(f"Gold evidence coverage: {coverage_count}/{total_expected} ({coverage_count/total_expected*100:.1f}%)")
            
    corpus = list(set(cache.values()))
    nei_retrieved_count = 0
    
    # For NEI claims, retrieve top-k from the corpus instead of passing empty evidence
    from src.evidence_retrieval import EvidenceIndex
    index = EvidenceIndex(backend=backend)
    if corpus:
        index.build_index(corpus)
        for ex in examples:
            if ex["label"] == "Unsupported" and not ex["gold_evidence"]:
                top_k_res = index.retrieve_evidence(ex["claim"], top_k=3)
                ex["gold_evidence"] = [r["text"] for r in top_k_res]
                ex["evidence_source"] = f"corpus retrieval (NEI, backend={index.active_backend})"
                nei_retrieved_count += 1
                
    stats = {
        "dataset_type": "Wikipedia extracts",
        "gold_evidence_missing": total_expected - coverage_count,
        "nei_evidence_retrieved": nei_retrieved_count,
        "total_samples": len(examples),
        "corpus_size": len(corpus),
        "retrieval_backend": index.active_backend if corpus else "none"
    }
    return (examples, corpus, stats) if return_stats else (examples, corpus)

if __name__ == "__main__":
    examples, corpus, stats = load_fever_subset(max_samples=10, seed=42, return_stats=True)
    print("=" * 80)
    print(f"FEVER Dataset Loader - Source: {stats['dataset_type']}")
    print(f"Total samples: {stats['total_samples']} | Corpus size: {stats['corpus_size']}")
    print(f"Counts: Gold missing = {stats['gold_evidence_missing']} | NEI candidates in copenlu = {stats.get('nei_candidate_sentences_present', 0)} | NEI retrieved from corpus = {stats.get('nei_evidence_retrieved_from_corpus', 0)}")
    print(f"Evidence Mode: {stats.get('evidence_mode', 'N/A')}")
    print("=" * 80)
    print("\nPrinting 5 Examples:\n")
    for i, ex in enumerate(examples[:5]):
        ev_sample = ex['gold_evidence'][0] if ex['gold_evidence'] else "[NO EVIDENCE]"
        print(f"Example {i+1}:")
        print(f"  Claim:           {ex['claim']}")
        print(f"  Label:           {ex['label']}")
        print(f"  Evidence Source: {ex['evidence_source']}")
        print(f"  Evidence Text:   {ev_sample[:120]}..." if len(ev_sample) > 120 else f"  Evidence Text:   {ev_sample}")
        print()

import re
import string
import urllib.request
import urllib.parse
import json

try:
    import numpy as np
except ImportError:
    np = None

from src.config import RetrievalConfig


class EvidenceIndex:
    def __init__(self, backend="dense", model_name=None, enable_wiki_fallback=True):
        self.backend = backend
        self.active_backend = backend
        self.docs = []
        self.doc_embeddings = None
        self.model = None
        self.enable_wiki_fallback = enable_wiki_fallback
        self._wiki_cache = {}
        
        if self.backend == "dense":
            if model_name is None:
                model_name = RetrievalConfig.dense_model_name
            try:
                if np is None:
                    raise ImportError("numpy is required for dense retrieval")
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.active_backend = "dense"
            except (ImportError, Exception):
                self.backend = "lexical"
                self.active_backend = "lexical_fallback"
                
    def build_index(self, docs: list[str]):
        if not docs:
            raise ValueError("Docs list is empty")
        self.docs = list(docs)
        if self.backend == "dense" and self.model is not None and np is not None:
            self.doc_embeddings = self.model.encode(self.docs)
            
    def _lexical_tokenize(self, text: str):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text.split()

    def _fetch_wikipedia_summary(self, entity: str) -> str | None:
        """
        Live Wikipedia REST API summary fetcher.
        Provides on-the-fly evidence retrieval for real-world entities.
        """
        if not entity or entity in self._wiki_cache:
            return self._wiki_cache.get(entity)

        clean_entity = entity.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_entity)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ClaimHallucinationDetector/1.0 (academic-nlp-demo; contact: research@example.edu)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    extract = data.get("extract", "").strip()
                    if extract:
                        self._wiki_cache[entity] = extract
                        return extract
        except Exception:
            pass

        self._wiki_cache[entity] = None
        return None

    def _extract_candidate_entities(self, claim: str) -> list[str]:
        """Extract multi-word proper nouns or title-cased entities from a claim."""
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        candidates = []
        for m in matches:
            if m.lower() not in {"the", "a", "an", "she", "he", "it", "they", "in", "on", "at"}:
                candidates.append(m)
        return candidates

    def retrieve_evidence(self, claim: str, top_k: int = 3) -> list[dict]:
        if not self.docs:
            raise ValueError("Index is empty")

        def _score_and_rank():
            if self.backend == "dense" and self.model is not None and np is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                claim_emb = self.model.encode([claim])
                sims = cosine_similarity(claim_emb, self.doc_embeddings)[0]
                top_indices = np.argsort(sims)[::-1][:top_k]
                return [{"text": self.docs[i], "score": float(sims[i])} for i in top_indices]
            else:
                claim_tokens = set(self._lexical_tokenize(claim))
                scored = []
                for doc in self.docs:
                    doc_tokens = set(self._lexical_tokenize(doc))
                    intersection = claim_tokens.intersection(doc_tokens)
                    union = claim_tokens.union(doc_tokens)
                    score = len(intersection) / len(union) if union else 0.0
                    scored.append((doc, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [{"text": doc, "score": float(score)} for doc, score in scored[:top_k]]

        results = _score_and_rank()
        top_score = results[0]["score"] if results else 0.0

        # Live Wikipedia Fallback if local retrieval confidence is low (< 0.15 lexical / < 0.35 dense)
        threshold = 0.35 if self.backend == "dense" else 0.15
        if self.enable_wiki_fallback and top_score < threshold:
            entities = self._extract_candidate_entities(claim)
            new_doc_added = False
            for ent in entities:
                summary = self._fetch_wikipedia_summary(ent)
                if summary and summary not in self.docs:
                    self.docs.append(summary)
                    new_doc_added = True
            if new_doc_added:
                if self.backend == "dense" and self.model is not None and np is not None:
                    self.doc_embeddings = self.model.encode(self.docs)
                results = _score_and_rank()

        return results

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        if self.active_backend == "dense" and self.model is not None and np is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            embs = self.model.encode([text_a, text_b])
            return float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        else:
            tokens_a = set(self._lexical_tokenize(text_a))
            tokens_b = set(self._lexical_tokenize(text_b))
            intersection = tokens_a.intersection(tokens_b)
            union = tokens_a.union(tokens_b)
            return len(intersection) / len(union) if union else 0.0

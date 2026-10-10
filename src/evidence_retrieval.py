import string
import numpy as np
from src.config import RetrievalConfig

class EvidenceIndex:
    def __init__(self, backend="dense", model_name=None):
        self.backend = backend
        self.docs = []
        self.doc_embeddings = None
        self.model = None
        
        if self.backend == "dense":
            if model_name is None:
                model_name = RetrievalConfig.dense_model_name
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except ImportError:
                print("sentence-transformers not installed, falling back to BM25")
                self.backend = "bm25"
                
    def build_index(self, docs: list[str]):
        if not docs:
            raise ValueError("Docs list is empty")
        self.docs = docs
        if self.backend == "dense" and self.model is not None:
            self.doc_embeddings = self.model.encode(docs)
            
    def _bm25_tokenize(self, text: str):
        # Strip punctuation
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text.split()

    def retrieve_evidence(self, claim: str, top_k: int = 3) -> list[dict]:
        if not self.docs:
            raise ValueError("Index is empty")
            
        if self.backend == "dense" and self.model is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            claim_emb = self.model.encode([claim])
            sims = cosine_similarity(claim_emb, self.doc_embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [{"text": self.docs[i], "score": float(sims[i])} for i in top_indices]
        else:
            # BM25 / Jaccard Fallback
            claim_tokens = set(self._bm25_tokenize(claim))
            scored = []
            for doc in self.docs:
                doc_tokens = set(self._bm25_tokenize(doc))
                intersection = claim_tokens.intersection(doc_tokens)
                union = claim_tokens.union(doc_tokens)
                score = len(intersection) / len(union) if union else 0.0
                scored.append((doc, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{"text": doc, "score": float(score)} for doc, score in scored[:top_k]]

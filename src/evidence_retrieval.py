"""
Evidence retrieval module.

Given an atomic claim, retrieves the most relevant supporting passages from
a source document or corpus. Planned approach for DA2: dense retrieval using
sentence-transformers embeddings, with optional BM25 fallback for exact-match
recall.
"""


def retrieve_evidence(claim: str, source_documents: list[str], top_k: int = 3) -> list[str]:
    """
    Retrieve top-k evidence passages relevant to a claim.

    Args:
        claim: The atomic claim to verify.
        source_documents: Candidate passages/documents to search.
        top_k: Number of top passages to return.

    Returns:
        List of retrieved evidence passages, ranked by relevance.
    """
    raise NotImplementedError("To be implemented in DA2.")

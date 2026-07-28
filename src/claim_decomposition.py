"""
Claim decomposition module.

Takes an LLM-generated response and decomposes it into atomic factual claims.
To be implemented in DA2 — planned approach: prompt-based decomposition using
an instruction-tuned LLM (following FActScore-style atomic fact extraction),
or a sentence-splitting + coreference-resolution pipeline as a lighter
alternative.
"""


def decompose_into_claims(response: str) -> list[str]:
    """
    Decompose a response into a list of atomic factual claims.

    Args:
        response: The LLM-generated text to decompose.

    Returns:
        List of atomic claim strings.
    """
    raise NotImplementedError("To be implemented in DA2.")

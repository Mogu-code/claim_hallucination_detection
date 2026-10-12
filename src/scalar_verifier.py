"""
Adaptive Scalar and Numerical Reasoning Verifier for DA3 Phase.
Augments cross-encoder NLI verification with rule-guided numerical and
scalar bound checking, directly addressing the #1 failure mode identified
in the FEVER/HaluEval error audit (e.g., scalar subsumption: 20 entails 5).
"""

import re
from typing import Optional, Tuple


# Word-to-number mapping for common English numerals
WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000
}


def extract_numbers_with_context(text: str) -> list[dict]:
    """
    Extract numbers (digits and words) and their immediately adjacent noun phrase.
    Returns a list of dicts: {"value": float, "context": str, "span": (start, end)}
    """
    results = []
    # 1. Match numeric digits (e.g. 5, 20, 1905, 3.5)
    digit_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\b')
    for m in digit_pattern.finditer(text):
        val = float(m.group(1))
        unit = m.group(2).lower() if m.group(2) else ""
        results.append({"value": val, "unit": unit, "raw": m.group(0)})

    # 2. Match number words (e.g. "five", "twenty")
    for word, val in WORD_NUMS.items():
        pattern = re.compile(rf'\b({word})\s+([a-zA-Z]+)\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            unit = m.group(2).lower() if m.group(2) else ""
            results.append({"value": float(val), "unit": unit, "raw": m.group(0)})

    return results


def check_scalar_subsumption(claim: str, evidence: str) -> Optional[Tuple[str, float]]:
    """
    Checks if evidence logically subsumes or contradicts the claim's numerical assertion.
    
    Returns:
        (label, boost_score) if a clear scalar relationship is found, else None.
        - ("entailment", 0.85): Evidence count >= Claim count in non-strict affirmative bound.
        - ("contradiction", 0.85): Direct numeric mismatch in exclusive contexts (e.g. years, exact dates).
    """
    claim_nums = extract_numbers_with_context(claim)
    ev_nums = extract_numbers_with_context(evidence)

    if not claim_nums or not ev_nums:
        return None

    # Check for unit / context matching
    for c_item in claim_nums:
        c_val = c_item["value"]
        c_unit = c_item["unit"]

        for e_item in ev_nums:
            e_val = e_item["value"]
            e_unit = e_item["unit"]

            # Same unit / noun context (e.g. "billionaires", "prizes", "years")
            if c_unit and e_unit and (c_unit == e_unit or c_unit.startswith(e_unit) or e_unit.startswith(c_unit)):
                # Case 1: Evidence value is strictly greater than or equal to claim value (subsumption)
                # E.g., Claim: "Yale has 5 billionaire alumni", Evidence: "Yale has 20 billionaire alumni"
                if e_val >= c_val and not any(k in claim.lower() for k in ["exactly", "only", "solely"]):
                    return ("entailment", 0.85)
                
                # Case 2: Claim asserts strictly more than evidence confirms
                # E.g., Claim: "Yale has 50 billionaire alumni", Evidence: "Yale has 20 billionaire alumni"
                if e_val < c_val:
                    return ("contradiction", 0.80)

            # Case 3: Temporal year bounds (e.g. "born in 1867" vs "born in 1905")
            if 1000 <= c_val <= 2050 and 1000 <= e_val <= 2050:
                if any(k in claim.lower() for k in ["born", "died", "established", "founded", "released", "in"]):
                    if any(k in evidence.lower() for k in ["born", "died", "established", "founded", "released", "in"]):
                        if abs(c_val - e_val) >= 1.0:
                            return ("contradiction", 0.90)
                        else:
                            return ("entailment", 0.90)

    return None

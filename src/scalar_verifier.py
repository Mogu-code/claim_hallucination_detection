"""
Adaptive Scalar and Numerical Reasoning Verifier for DA3 Phase.
Augments cross-encoder NLI verification with rule-guided numerical and
scalar bound checking (e.g., scalar subsumption: 20 entails 5).
Marked as an exploratory verification feature.
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

STOP_WORDS = {
    "in", "on", "at", "to", "from", "of", "by", "for", "with", "about", "as",
    "into", "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "and", "or", "but", "the", "a",
    "an"
}


def normalize_noun(w: str) -> str:
    w = w.lower()
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def extract_numbers_with_context(text: str) -> list[dict]:
    """
    Extract numbers (digits and words) and their immediately adjacent unit noun.
    Returns a list of dicts: {"value": float, "unit": str}
    """
    results = []
    # 1. Match numeric digits (e.g. 5, 20, 1905, 3.5)
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
        val = float(m.group(1))
        after = text[m.end():]
        words_match = re.match(r'^\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', after)
        unit = ""
        if words_match:
            raw_words = words_match.group(1).split()
            valid_words = []
            for w in raw_words:
                if w.lower() in STOP_WORDS:
                    break
                valid_words.append(normalize_noun(w))
            unit = " ".join(valid_words)
        results.append({"value": val, "unit": unit})

    # 2. Match number words (e.g. "one", "five", "twenty")
    for word, val in WORD_NUMS.items():
        for m in re.finditer(rf'\b({word})\b', text, re.IGNORECASE):
            after = text[m.end():]
            words_match = re.match(r'^\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', after)
            unit = ""
            if words_match:
                raw_words = words_match.group(1).split()
                valid_words = []
                for w in raw_words:
                    if w.lower() in STOP_WORDS:
                        break
                    valid_words.append(normalize_noun(w))
                unit = " ".join(valid_words)
            results.append({"value": float(val), "unit": unit})

    return results


def check_scalar_subsumption(claim: str, evidence: str) -> Optional[Tuple[str, float]]:
    """
    Checks if evidence logically subsumes or contradicts the claim's numerical assertion.
    Requires matching unit nouns for number comparisons.
    Does NOT match generic years without unit nouns.
    
    Returns:
        (label, boost_score) if a clear scalar relationship is found, else None.
        - ("entailment", 0.85): Evidence count >= Claim count for identical unit noun.
        - ("contradiction", 0.80): Claim count > Evidence count for identical unit noun.
    """
    claim_nums = extract_numbers_with_context(claim)
    ev_nums = extract_numbers_with_context(evidence)

    if not claim_nums or not ev_nums:
        return None

    # Check for identical unit / noun context
    for c_item in claim_nums:
        c_val = c_item["value"]
        c_unit = c_item["unit"]

        # Number comparisons require a non-empty unit noun
        if not c_unit:
            continue

        for e_item in ev_nums:
            e_val = e_item["value"]
            e_unit = e_item["unit"]

            if not e_unit:
                continue

            # Require the same unit noun (exact match or matching head noun)
            same_unit = (c_unit == e_unit) or (
                bool(c_unit.split() and e_unit.split()) and (c_unit.split()[-1] == e_unit.split()[-1])
            )

            if same_unit:
                # Case 1: Evidence value is strictly greater than or equal to claim value (subsumption)
                # E.g., Claim: "Yale has 5 living billionaire alumni", Evidence: "Yale has 20 living billionaire alumni"
                if e_val >= c_val and not any(k in claim.lower() for k in ["exactly", "only", "solely"]):
                    return ("entailment", 0.85)
                
                # Case 2: Claim asserts strictly more than evidence confirms
                # E.g., Claim: "Yale has 50 living billionaire alumni", Evidence: "Yale has 20 living billionaire alumni"
                if e_val < c_val:
                    return ("contradiction", 0.80)

    return None

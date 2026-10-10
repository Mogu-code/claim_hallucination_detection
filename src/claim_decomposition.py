import re

def decompose_into_claims(response: str) -> list[str]:
    """
    Decompose a response into a list of atomic factual claims.
    """
    if not response or not response.strip():
        return []
        
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
    claims = []
    
    last_subject = ""
    
    for sentence in sentences:
        if not sentence: continue
        
        # Split on conjunctions
        parts = re.split(r',?\s+(?:and|but)\s+', sentence)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part: continue
            
            words = part.split()
            first_word = words[0]
            
            # Resolve leading pronoun
            if first_word.lower() in ["he", "she", "it", "they"]:
                if last_subject:
                    part = last_subject + " " + " ".join(words[1:])
                    words = part.split()
            # Carry over subject for verb-initial second clauses
            elif i > 0 and first_word.lower() not in ["he", "she", "it", "they"] and first_word.islower():
                if last_subject:
                    part = last_subject + " " + part
                    words = part.split()
                    
            claims.append(part)
            
            # Update subject based on the first part
            if i == 0:
                subj_tokens = []
                for w in words:
                    if w.istitle() or w.lower() in ["the", "a", "an"]:
                        subj_tokens.append(w)
                    else:
                        break
                if subj_tokens:
                    last_subject = " ".join(subj_tokens)
                elif last_subject == "":
                    last_subject = first_word

    return claims

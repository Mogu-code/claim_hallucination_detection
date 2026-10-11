import re

def decompose_into_claims(response: str) -> list[str]:
    """
    Decompose a response into a list of atomic factual claims.
    """
    if not response or not response.strip():
        return []
        
    sentence_pattern = re.compile(r'(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\bU\.S\.)(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_pattern.split(response) if s.strip()]
    claims = []
    
    last_subject = ""
    
    for sentence in sentences:
        if not sentence: continue
        
        # Split on conjunctions
        parts = re.split(r',?\s+(and|but)\s+', sentence)
        
        merged_parts = [parts[0].strip()]
        for i in range(1, len(parts), 2):
            conj = parts[i]
            next_part = parts[i+1].strip()
            
            left_words = merged_parts[-1].split()
            right_words = next_part.split()
            
            should_split = False
            if len(left_words) >= 3 and len(right_words) >= 3:
                first_right = right_words[0]
                if first_right.islower() and first_right not in ["the", "a", "an"]:
                    should_split = True
                    
            if should_split:
                merged_parts.append(next_part)
            else:
                merged_parts[-1] = merged_parts[-1] + " " + conj + " " + next_part
                
        for i, part in enumerate(merged_parts):
            part = part.strip()
            if not part: continue
            
            words = part.split()
            if not words: continue
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
                    
            if len(words) > 1:
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

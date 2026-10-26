# FEVER Evaluation: Qualitative Error Analysis (A5 Full System)

**Course:** BCSE306L — Natural Language Processing  
**Dataset:** FEVER (`copenlu/fever_gold_evidence`, Dev Test Split, $N=500$, Seed=42)  
**Evaluated Model:** $A_5$ Full System (NLI + Dense Retrieval Similarity Fusion)  
**Thresholds Applied:** $\tau_{\text{ent}} = 0.55$, $\tau_{\text{contra}} = 0.30$, $\tau_{\text{sim}} = 0.05$ (tuned on disjoint 200-sample tune split with extended grid)

---

## 1. Quantitative Performance Overview

On the disjoint 500-sample FEVER test split, the full fusion system ($A_5$) achieved:
- **Accuracy:** 83.40% [95% Bootstrap CI: 80.00%, 86.60%]
- **Macro-F1:** 0.8361 [95% Bootstrap CI: 0.8015, 0.8678]
- **Total Test Examples:** 500
- **Correct Predictions:** 417 (83.4%)
- **Total Misclassifications:** 83 (16.6%)

The confusion matrix below highlights the distribution of these 83 misclassifications across the three classes (**Supported**, **Contradicted**, and **Unsupported**):

![Confusion Matrix of A5 Full System](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/confusion_matrix_a5.png)

```
                       Predicted Label
                 Supported  Contradicted  Unsupported
True Supported       122          5            26
True Contradicted      4        137            15
True Unsupported       9         24           158
```

Key observations from the confusion matrix:
1. **False Unsupported Errors ($Supported \to Unsupported$, $N=26$):** The largest single source of error occurs when genuine entailment fails to cross the calibrated entailment threshold ($\tau_{\text{ent}} = 0.55$) or the semantic similarity gate ($\tau_{\text{sim}} = 0.05$) due to paraphrasing or numerical reasoning.
2. **Unsupported vs. Contradicted Confusion ($Unsupported \to Contradicted$, $N=24$):** Negative facts or geographical mutual exclusion (e.g., being Governor of Texas vs. Maine) lead the NLI model to infer a contradiction, whereas human FEVER annotators strictly labeled the claim as *Not Enough Info* (Unsupported) due to the absence of explicit disproof.
3. **Subtle Refutation Misses ($Contradicted \to Unsupported$, $N=15$):** Complex negations, implicit role distinctions, or missing contextual dates fail to trigger the contradiction threshold ($\tau_{\text{contra}} = 0.30$).

---

## 2. Categorization of 20 Representative Error Cases

The 20 misclassified cases extracted directly from the test split ([experiments/error_cases.json](file:///c:/Users/MOKSHDAA/Desktop/github/claim-hallucination-detection/experiments/error_cases.json)) are grouped into five distinct root-cause taxonomies:

### Taxonomy 1: Numerical Reasoning & Scalar Bounds (False Unsupported) — 2 Cases (10.0%)
The NLI model struggles with downward entailment and scalar quantities where a larger number logically entails a smaller subset.

* **Case 1 (Index 8):**
  * **Claim:** *"Yale University's alumni includes five billionaires."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"Yale has graduated many notable alumni , including five U.S. Presidents , 19 U.S. Supreme Court Justices , 20 living billionaires , and many heads of state ."*
  * **Scores:** $P(\text{ent}) = 0.2913$, $P(\text{contra}) = 0.1296$, $\text{Sim} = 0.8230$
  * **Root Cause:** DeBERTa treats the scalar discrepancy ("20 living billionaires" vs. "five billionaires") as a neutral mismatch rather than realizing that 20 living billionaires logically subsumes five ($20 \ge 5$). Because $P(\text{ent}) < 0.55$, it was classified as Unsupported.

* **Case 2 (Index 87):**
  * **Claim:** *"Matthias Corvinus had one of the largest collections of books in Europe during the Renaissance in the 1450's."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"Matthias patronized art and science ; his royal library , the Bibliotheca Corviniana , was one of the largest collections of books in Europe ."*
  * **Scores:** $P(\text{ent}) = 0.2603$, $P(\text{contra}) = 0.0045$, $\text{Sim} = 0.7571$
  * **Root Cause:** The evidence confirms the collection was one of Europe's largest, but omits the exact decade ("1450's"). The model penalizes the missing temporal token, yielding $P(\text{ent}) = 0.2603 < 0.55$.

---

### Taxonomy 2: Paraphrasing & Lexical Abstraction (False Unsupported) — 4 Cases (20.0%)
Claims that require high-level abstraction or non-literal semantic inference fail to generate sufficient direct entailment probability.

* **Case 3 (Index 17):**
  * **Claim:** *"Bad Romance was profitable around the world."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"It achieved worldwide success by topping the charts in a variety of markets , ultimately selling 12 million copies worldwide , thus becoming one of the best-selling singles of all time ."*
  * **Scores:** $P(\text{ent}) = 0.2617$, $P(\text{contra}) = 0.0052$, $\text{Sim} = 0.3762$
  * **Root Cause:** Lexical gap between the abstract concept *"profitable"* and the factual metric *"selling 12 million copies worldwide"*. The NLI backbone assigns high neutral mass ($0.733$) because commercial profit is not explicitly stated.

* **Case 4 (Index 115):**
  * **Claim:** *"Liverpool was where The Beatles came together."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"The popularity of The Beatles and other groups from the Merseybeat era contributes to Liverpool 's status as a tourist destination ."*
  * **Scores:** $P(\text{ent}) = 0.3386$, $P(\text{contra}) = 0.0109$, $\text{Sim} = 0.7519$
  * **Root Cause:** Human annotators relied on external world knowledge that Merseybeat implies band formation in Liverpool. DeBERTa strictly checks text-level entailment and correctly notes that tourism status does not explicitly prove geographic origin ($P(\text{ent}) = 0.3386 < 0.40$).

* **Case 5 (Index 68):**
  * **Claim:** *"James VI and I was a major advocate of a single parliament for England."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"He was a major advocate of a single parliament for England and Scotland ."*
  * **Scores:** $P(\text{ent}) = 0.0666$, $P(\text{contra}) = 0.0100$, $\text{Sim} = 0.7747$
  * **Root Cause:** The conjunction *"England and Scotland"* leads DeBERTa to treat the claim as incomplete/neutral rather than supported, suppressing the entailment score to $0.0666$.

* **Case 6 (Index 85):**
  * **Claim:** *"Henry VIII (TV serial) features an English actor in a leading role."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"Raymond Andrew `` Ray '' Winstone born 19 February 1957 is an English film and television actor ."*
  * **Scores:** $P(\text{ent}) = 0.0660$, $P(\text{contra}) = 0.0026$, $\text{Sim} = 0.3803$
  * **Root Cause:** Multi-hop reasoning failure: the evidence proves Ray Winstone is an English actor, but the connection between Ray Winstone and the TV serial *Henry VIII* was in an unretrieved sentence.

---

### Taxonomy 3: World Knowledge & Geographic Exclusion (Unsupported $\to$ Contradicted) — 5 Cases (25.0%)
When candidate evidence mentions an entity's real property, the model infers mutual exclusivity with the claim's property and outputs Contradiction, whereas FEVER guidelines classify it as Unsupported (NEI).

* **Case 7 (Index 1):**
  * **Claim:** *"WGBH-TV is located in the capital of Maine."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"WGBX 's studios are located on Guest Street in Boston , and its transmitter is located in Needham , Massachusetts ."*
  * **Scores:** $P(\text{ent}) = 0.0041$, $P(\text{contra}) = 0.8774$, $\text{Sim} = 0.5951$
  * **Root Cause:** Because the evidence shows the transmitter is in Massachusetts and Boston is not the capital of Maine (Augusta is), DeBERTa outputs strong contradiction ($0.8774$). FEVER treats this as Unsupported because WGBH-TV's relationship to Maine is technically unaddressed.

* **Case 8 (Index 59):**
  * **Claim:** *"Ann Richards was the head of the executive branch of Maine's government."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"Ann Richards was an American politician and the 45th Governor of Texas ."*
  * **Scores:** $P(\text{ent}) = 0.0293$, $P(\text{contra}) = 0.4333$, $\text{Sim} = 0.6042$
  * **Root Cause:** Being the Governor of Texas implies mutual exclusivity with heading Maine's government. $P(\text{contra}) = 0.4333 \ge 0.40$ triggers Contradicted.

* **Case 9 (Index 75):**
  * **Claim:** *"Asylum Records is an American record label founded in 1971 by David Geffen and musician Elliot Roberts."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"Roberts later formed Lookout Management with David Geffen , and helped to create Geffen 's Asylum Records in 1970 , which merged with Elektra Records in 1972..."*
  * **Scores:** $P(\text{ent}) = 0.0503$, $P(\text{contra}) = 0.8521$, $\text{Sim} = 0.6654$
  * **Root Cause:** The year 1970 in the evidence directly contradicts 1971 in the claim ($P(\text{contra}) = 0.8521$). FEVER gold annotation marked this NEI/Unsupported due to annotation disagreement.

* **Case 10 (Index 31):**
  * **Claim:** *"Earl Scruggs was an actor"*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"Beginning with his first recordings with Bill Monroe and His Blue Grass Boys , and later with Lester Flatt , Earl Scruggs and the Foggy Mountain Boys , Earl Scruggs introduced a vocabulary of `` licks '' , short musical phrases that are reused in many different songs ."*
  * **Scores:** $P(\text{ent}) = 0.0081$, $P(\text{contra}) = 0.7974$, $\text{Sim} = 0.5728$
  * **Root Cause:** DeBERTa considers professional identity as a musician mutually exclusive with actor ($P(\text{contra}) = 0.7974 \ge 0.30$).

* **Case 11 (Index 125):**
  * **Claim:** *"Caroline Kennedy served as Japanese Ambassador to the United States."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"...Kennedy and her uncle Ted endorsed Democratic candidate Barack Obama for President ; she later stumped for him... and addressed the 2008 Democratic National Convention..."*
  * **Scores:** $P(\text{ent}) = 0.0056$, $P(\text{contra}) = 0.6128$, $\text{Sim} = 0.3571$
  * **Root Cause:** Over-prediction of contradiction from non-matching biographical roles.

---

### Taxonomy 4: Complex Negation & Subtle Contradiction (False Unsupported) — 4 Cases (20.0%)
The claim negates an event or asserts non-involvement, but the evidence phrases the relationship positively without overt antonyms, leaving contradiction probability under threshold.

* **Case 12 (Index 12):**
  * **Claim:** *"The Hundred Years' War does not include the Edwardian Era War."*
  * **Gold Label:** `Contradicted` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"Historians commonly divide the war into three phases separated by truces : the Edwardian Era War -LRB- 1337 -- 1360 -RRB- ; the Caroline War ; and the Lancastrian War ."*
  * **Scores:** $P(\text{ent}) = 0.2671$, $P(\text{contra}) = 0.3303$, $\text{Sim} = 0.4359$
  * **Root Cause:** The evidence lists the Edwardian Era War as one of three phases. However, due to complex clausal negation ("does not include"), DeBERTa assigns only $0.3303$ to contradiction, narrowly missing the $\tau_{\text{contra}} = 0.40$ threshold.

* **Case 13 (Index 48):**
  * **Claim:** *"Sheryl Lee has yet to appear in a film as of 2016."*
  * **Gold Label:** `Contradicted` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"Notable film roles include as Astrid Kirchherr in Backbeat -LRB- 1994 -RRB- , as well as a lead role in the drama Mother Night -LRB- 1996 -RRB-..."*
  * **Scores:** $P(\text{ent}) = 0.0028$, $P(\text{contra}) = 0.0276$, $\text{Sim} = 0.1770$
  * **Root Cause:** Low lexical similarity ($\text{Sim} = 0.1770$) between "has yet to appear" and film role listings from the 1990s suppresses attention weights in the cross-encoder.

* **Case 14 (Index 93):**
  * **Claim:** *"Entourage (film) was a film adaptation of a comic."*
  * **Gold Label:** `Contradicted` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"It serves as a continuation of the HBO TV series of the same name created by Ellin ."*
  * **Scores:** $P(\text{ent}) = 0.0384$, $P(\text{contra}) = 0.0162$, $\text{Sim} = 0.1764$
  * **Root Cause:** Being a continuation of a TV series does not strictly rule out a comic book foundation in formal first-order logic; DeBERTa treats this as neutral ($0.945$).

* **Case 15 (Index 119):**
  * **Claim:** *"Paul Nicholls created Joe Wicks."*
  * **Gold Label:** `Contradicted` | **Predicted Label:** `Unsupported`
  * **Evidence:** *"He is best known for his roles as Joe Wicks in BBC 's EastEnders , and as DS Sam Casey in ITV 's Law & Order : UK ."*
  * **Scores:** $P(\text{ent}) = 0.0424$, $P(\text{contra}) = 0.0339$, $\text{Sim} = 0.3615$
  * **Root Cause:** Actor playing a character vs. creator: DeBERTa does not perceive the semantic role incompatibility ("played" vs. "created") as a direct contradiction ($P(\text{contra}) = 0.0339$).

---

### Taxonomy 5: Entity Polysemy & Retrieval Distractor Mismatches — 5 Cases (25.0%)
Dense retrieval retrieves an entity with the same surface form but an entirely different semantic category, corrupting downstream NLI reasoning.

* **Case 16 (Index 129):**
  * **Claim:** *"House is a sitcom."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"A house may have a separate dining room , or the eating area may be integrated into another room ."*
  * **Scores:** $P(\text{ent}) = 0.0037$, $P(\text{contra}) = 0.6567$, $\text{Sim} = 0.3825$
  * **Root Cause:** Polysemy collision: The claim refers to the TV show *House (MD)*, but dense retrieval retrieved the Wikipedia page for a residential dwelling. Describing room partitions contradicts being a sitcom ($0.6567$).

* **Case 17 (Index 83):**
  * **Claim:** *"Akkineni Nageswara Rao shot Daag."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"Vedam Jaishankar , cricket correspondent"*
  * **Scores:** $P(\text{ent}) = 0.0066$, $P(\text{contra}) = 0.8530$, $\text{Sim} = 0.3818$
  * **Root Cause:** Unrelated distractor sentence retrieved due to high TF-IDF/dense matching on rare entity tokens; DeBERTa assigns high contradiction to completely unrelated claims.

* **Case 18 (Index 51):**
  * **Claim:** *"Vedam is a 2010 Indian drama film."*
  * **Gold Label:** `Supported` | **Predicted Label:** `Contradicted`
  * **Evidence:** *"The film was dubbed in Malayalam and Krish later remade the film in Tamil as Vaanam (2011)..."*
  * **Scores:** $P(\text{ent}) = 0.0618$, $P(\text{contra}) = 0.6270$, $\text{Sim} = 0.4882$
  * **Root Cause:** The evidence sentence describes the 2011 Tamil remake *Vaanam*; the model flags the 2010 vs. 2011 year mismatch as a contradiction rather than noting it is a subsequent remake.

* **Case 19 (Index 37):**
  * **Claim:** *"Kenneth Lonergan was born in New York."*
  * **Gold Label:** `Unsupported` | **Predicted Label:** `Supported`
  * **Evidence:** *"Kenneth Lonergan -LRB- born 1962 -RRB- , New York playwright , screenwriter , and director"*
  * **Scores:** $P(\text{ent}) = 0.5957$, $P(\text{contra}) = 0.0023$, $\text{Sim} = 0.6990$
  * **Root Cause:** Spurious correlation: "New York playwright" does not strictly prove birth in New York, but DeBERTa associates geographical adjectives directly with the birth location ($P(\text{ent}) = 0.5957 \ge 0.40$).

* **Case 20 (Index 46):**
  * **Claim:** *"Baz Luhrmann's film Australia is solely an epic historical romantic comedy."*
  * **Gold Label:** `Supported` (FEVER claim formulation) | **Predicted Label:** `Contradicted`
  * **Evidence:** *"His 2008 film Australia is an epic historical romantic drama film starring Hugh Jackman and Nicole Kidman ."*
  * **Scores:** $P(\text{ent}) = 0.0279$, $P(\text{contra}) = 0.7871$, $\text{Sim} = 0.5061$
  * **Root Cause:** The word "drama" in evidence directly contradicts "comedy" in the claim ($P(\text{contra}) = 0.7871$). The annotation in FEVER was flawed or misaligned with the qualifier "solely".

---

## 3. Summary of Architectural Recommendations for DA3

1. **Adaptive Scalar/Numerical Verifier:** Augment the cross-encoder with a dedicated numeric parser for comparative claims ($X \ge Y$) to resolve cases like Yale's billionaires ($20 \ge 5$).
2. **Entity Disambiguation Prior to Retrieval:** Use named entity linking (e.g., Wikidata ID matching) to prevent catastrophic polysemy errors (such as physical dwelling *House* vs. TV drama *House*).
3. **Threshold Decoupling for NEI Claims:** Establish a separate neutral-confidence zone to prevent mutually exclusive geographical/biographical statements from defaulting to Contradiction when the reference sentence simply lacks relevant information.

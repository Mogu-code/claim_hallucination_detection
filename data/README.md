# Datasets

Raw data is **not** committed to this repo (see `.gitignore`). Use the loader
scripts in `loaders/` to fetch data locally.

## HaluEval (primary)

- Source: https://github.com/RUCAIBox/HaluEval
- Also available via HuggingFace `datasets`: `pminervini/HaluEval` (or load
  directly from the raw JSON files in the original repo — see loader script)
- Splits used: QA, dialogue, summarization
- License: check original repo before redistribution

## FEVER (secondary)

- Source: https://fever.ai/dataset/fever.html
- Used for the Supported / Refuted / NotEnoughInfo verification paradigm
- Access: free registration required for full wiki dump; claim/label pairs
  available via HuggingFace `datasets`: `fever`

## Access status

- [ ] HaluEval downloaded and verified locally
- [ ] FEVER downloaded and verified locally
- [ ] Sample subset extracted for early prototyping

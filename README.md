# dissertation-appendix
This repository contains the code and data for the empirical analysis conducted as part of this dissertation. The study examines whether incentive framing in system prompts elicits dark pattern behaviours in open-source large language models.

## Repository Structure

- `data/`
  - `raw/` — DarkBench prompts and validation sample
  - `generated/` — Model outputs (3,960 responses)
  - `judged/` — Majority-voted judge labels
  - `lexical/` — NRC-EIL and politeness feature CSVs
- `results/`
  - `tables/` — Summary tables
- `scripts/`
  - `generation/` — Response generation
  - `judging/` — LLM-as-judge pipeline
  - `lexical/` — Emotion and politeness analysis
  - `stats/` — Statistical analysis
- `requirements.txt`


## Models Used
The following open-source models were used, all accessed via HuggingFace:

- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

Note: Llama-3.1-8B-Instruct requires explicit licence approval from Meta before access is granted on HuggingFace.



## Data

All generated outputs, judged labels, and lexical features are included in this repository. The DarkBench prompts are included for reproducibility. The validation sample (`data/raw/validation/validation_sample_150.csv`) contains 150 manually annotated responses used for judge validation.


## Lexicon

This project uses the **NRC Emotion Intensity Lexicon (NRC-EIL) v1**, created by 
Dr. Saif M. Mohammad at the National Research Council Canada (NRC).

The lexicon is **not included in this repository** due to its licensing terms 
(non-commercial use only, no redistribution).

To reproduce the lexical analysis:
1. Download the lexicon from: https://saifmohammad.com/WebPages/AffectIntensity.htm
2. Place the file at: `data/raw/NRC-Emotion-Intensity-Lexicon-v1.txt`

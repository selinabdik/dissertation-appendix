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

## Pipeline

Run the scripts in the following order:

**1. Generate model outputs**
```bash
python scripts/generation/generate.py --model mistral --prompts data/raw/darkbench_prompts/darkbench_prompts.csv --output data/generated/mistral_outputs.json
python scripts/generation/generate.py --model llama --prompts data/raw/darkbench_prompts/darkbench_prompts.csv --output data/generated/llama_outputs.json
python scripts/generation/generate.py --model qwen --prompts data/raw/darkbench_prompts/darkbench_prompts.csv --output data/generated/qwen_outputs.json
```

**2. Judge outputs**
```bash
python scripts/judging/judge_mistral.py --input data/generated/mistral_outputs.json --output data/judged/mistral_judged.json
python scripts/judging/judge_llama.py --input data/generated/llama_outputs.json --output data/judged/llama_judged.json
python scripts/judging/judge_qwen.py --input data/generated/qwen_outputs.json --output data/judged/qwen_judged.json
```

**3. Majority vote**
```bash
python scripts/judging/majority_vote.py --judge1 data/judged/mistral_judged.json --judge2 data/judged/llama_judged.json --judge3 data/judged/qwen_judged.json --output data/judged/mistral_final.json
```

**4. Lexical analysis** (run from `scripts/lexical/`)
```bash
cd scripts/lexical
python nrceil_analysis.py
python politeness_analysis.py
python analysis.py
```

**5. Statistical analysis** (run from `scripts/stats/`)
```bash
cd scripts/stats
python stats_block1.py
```

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

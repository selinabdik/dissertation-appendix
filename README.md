# dissertation-appendix
Code and data for my dissertation on dark patterns in LLMs.

# Dark Patterns in Open-Source LLMs

## Overview
This repository contains the code and data for my BSc dissertation:
**"Investigating Dark Pattern Behaviours in Open-Source Large Language 
Models Under Varying Incentive Conditions"**

## Repository Structure
- `data/` — raw inputs, generated outputs, judged labels, lexical scores
- `scripts/` — Python scripts for generation, judging, lexical analysis, 
  and statistics
- `results/` — output tables and figures

## Reproducing the Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Generate outputs
```bash
python scripts/generation/generate.py
```

### 3. Run judge pipeline
```bash
python scripts/judging/judge_mistral.py
python scripts/judging/judge_llama.py
python scripts/judging/judge_qwen.py
python scripts/judging/majority_vote.py
```

### 4. Run lexical analysis
```bash
python scripts/lexical/politeness_analysis.py
python scripts/lexical/nrceil_analysis.py
python scripts/lexical/analysis.py
```

### 5. Run statistical tests
```bash
python scripts/stats/stats_block1.py
python scripts/stats/inferential_stats.py
```

## Models Used
- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

## Lexicon

This project uses the **NRC Emotion Intensity Lexicon (NRC-EIL) v1**, created by 
Dr. Saif M. Mohammad at the National Research Council Canada (NRC).

The lexicon is **not included in this repository** due to its licensing terms 
(non-commercial use only, no redistribution).

To reproduce the lexical analysis:
1. Download the lexicon from: https://saifmohammad.com/WebPages/AffectIntensity.htm
2. Place the file at: `data/raw/NRC-Emotion-Intensity-Lexicon-v1.txt`

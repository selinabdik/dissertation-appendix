import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

BASE = '/Users/selinabdik/Desktop/dissertation_outputs'
MODELS = ['mistral', 'llama', 'qwen']
LABELS = ['sycophancy', 'anthropomorphisation', 'user_retention']
CONDITIONS = ['neutral', 'engagement', 'persuasion', 'trustworthiness']
CATEGORIES = ['Sycophancy', 'Anthropomorphization', 'User retention']

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def cramers_v(chi2, n, k):
    return np.sqrt(chi2 / (n * (min(k, 2) - 1)))

# ============================================================
# SECTION 1: CHI-SQUARE BY CONDITION
# ============================================================
print("\n" + "="*60)
print("SECTION 1: DARK PATTERN RATES BY CONDITION")
print("="*60)

for model in MODELS:
    with open(f'{BASE}/{model}_final.json') as f:
        data = json.load(f)
    print(f'\n{model.upper()}:')
    for label in LABELS:
        contingency = []
        for c in CONDITIONS:
            subset = [d for d in data if d['condition'] == c]
            pos = sum(1 for d in subset if d[label] == 1)
            neg = len(subset) - pos
            contingency.append([pos, neg])
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = sum(sum(row) for row in contingency)
        v = cramers_v(chi2, n, len(contingency))
        print(f'  {label}: chi2={chi2:.3f}, df={dof}, p={p:.4f} {sig_stars(p)}, V={v:.3f}')

# ============================================================
# SECTION 2: CHI-SQUARE BY MODEL
# ============================================================
print("\n" + "="*60)
print("SECTION 2: DARK PATTERN RATES BY MODEL")
print("="*60)

all_data = {}
for model in MODELS:
    with open(f'{BASE}/{model}_final.json') as f:
        all_data[model] = json.load(f)

for label in LABELS:
    contingency = []
    for model in MODELS:
        pos = sum(1 for d in all_data[model] if d[label] == 1)
        neg = len(all_data[model]) - pos
        contingency.append([pos, neg])
    chi2, p, dof, _ = chi2_contingency(contingency)
    n = sum(sum(row) for row in contingency)
    v = cramers_v(chi2, n, len(contingency))
    print(f'{label}: chi2={chi2:.3f}, df={dof}, p={p:.4f} {sig_stars(p)}, V={v:.3f}')

# ============================================================
# SECTION 3: CHI-SQUARE BY PROMPT CATEGORY
# ============================================================
print("\n" + "="*60)
print("SECTION 3: DARK PATTERN RATES BY PROMPT CATEGORY")
print("="*60)

for model in MODELS:
    with open(f'{BASE}/{model}_final.json') as f:
        data = json.load(f)
    print(f'\n{model.upper()}:')
    for label in LABELS:
        contingency = []
        for cat in CATEGORIES:
            subset = [d for d in data if d['category'] == cat]
            pos = sum(1 for d in subset if d[label] == 1)
            neg = len(subset) - pos
            contingency.append([pos, neg])
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = sum(sum(row) for row in contingency)
        v = cramers_v(chi2, n, len(contingency))
        print(f'  {label}: chi2={chi2:.3f}, df={dof}, p={p:.4f} {sig_stars(p)}, V={v:.3f}')

# ============================================================
# SECTION 4: OVERALL COHEN'S KAPPA
# ============================================================
print("\n" + "="*60)
print("SECTION 4: OVERALL COHEN'S KAPPA")
print("="*60)

df_val = pd.read_csv(f'{BASE}/validation_sample_150.csv')
df_val = df_val.dropna(subset=['human_sycophancy','human_anthropomorphisation','human_user_retention'])
print(f'Total annotated: {len(df_val)}')

for label in LABELS:
    judge = df_val[f'judge_{label}'].astype(int)
    human = df_val[f'human_{label}'].astype(int)
    if len(set(judge)) > 1 and len(set(human)) > 1:
        kappa = cohen_kappa_score(human, judge)
        agreement = (judge == human).mean()
        print(f'{label}: kappa={kappa:.3f}, agreement={agreement:.1%}, judge_pos={judge.sum()}, human_pos={human.sum()}')
    else:
        print(f'{label}: insufficient variance for Kappa')

print("\nDone.")
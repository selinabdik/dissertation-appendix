import pandas as pd
import numpy as np
from scipy import stats
import json

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

def mannwhitney(df, label, feature):
    g0 = df[df[label]==0][feature].values
    g1 = df[df[label]==1][feature].values
    if len(g0) == 0 or len(g1) == 0:
        return None, None, None
    stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    d = cohens_d(g1, g0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    return p, sig, d

BASE = "../data"
MODELS = ['mistral', 'llama', 'qwen']
LABELS = ['sycophancy', 'anthropomorphisation', 'user_retention']
CONDITIONS = ['neutral', 'engagement', 'persuasion', 'trustworthiness']
CATEGORIES = ['Sycophancy', 'Anthropomorphization', 'User retention']

# ============================================================
# SECTION 1 — OVERALL DARK PATTERN RATES BY MODEL
# ============================================================

print("\n" + "="*60)
print("SECTION 1 — OVERALL DARK PATTERN RATES BY MODEL")
print("="*60)

for model in MODELS:
    with open(f"{BASE}/{model}_final.json") as f:
        data = json.load(f)
    print(f"\n{model.upper()} (n={len(data)}):")
    for label in LABELS + ['invalid']:
        pos = sum(1 for d in data if d[label] == 1)
        print(f"  {label}: {pos}/{len(data)} ({100*pos/len(data):.1f}%)")

# ============================================================
# SECTION 2 — DARK PATTERN RATES BY INCENTIVE CONDITION
# ============================================================

print("\n" + "="*60)
print("SECTION 2 — DARK PATTERN RATES BY INCENTIVE CONDITION")
print("="*60)

for model in MODELS:
    with open(f"{BASE}/{model}_final.json") as f:
        data = json.load(f)
    print(f"\n{model.upper()}:")
    for condition in CONDITIONS:
        subset = [d for d in data if d['condition'] == condition]
        print(f"  {condition.upper()} (n={len(subset)}):")
        for label in LABELS:
            pos = sum(1 for d in subset if d[label] == 1)
            print(f"    {label}: {pos}/{len(subset)} ({100*pos/len(subset):.1f}%)")

# ============================================================
# SECTION 3 — DARK PATTERN RATES BY PROMPT CATEGORY
# ============================================================

print("\n" + "="*60)
print("SECTION 3 — DARK PATTERN RATES BY PROMPT CATEGORY")
print("="*60)

for model in MODELS:
    with open(f"{BASE}/{model}_final.json") as f:
        data = json.load(f)
    print(f"\n{model.upper()}:")
    for category in CATEGORIES:
        subset = [d for d in data if d['category'] == category]
        print(f"  {category.upper()} PROMPTS (n={len(subset)}):")
        for label in LABELS:
            pos = sum(1 for d in subset if d[label] == 1)
            print(f"    {label}: {pos}/{len(subset)} ({100*pos/len(subset):.1f}%)")

# ============================================================
# SECTION 4 — POLITENESS FEATURES BY DARK PATTERN PRESENCE
# ============================================================

print("\n" + "="*60)
print("SECTION 4 — POLITENESS FEATURES BY DARK PATTERN PRESENCE")
print("="*60)

KEY_POL = ['Hedges', '1st_person', '2nd_person', 'Gratitude',
           'Deference', 'HASPOSITIVE', 'total_politeness_strategies']

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_politeness.csv")
    print(f"\n{model.upper()}:")
    for label in LABELS:
        print(f"\n  {label}:")
        print(df.groupby(label)[KEY_POL].mean().round(3).to_string())

# ============================================================
# SECTION 5 — POLITENESS FEATURES BY INCENTIVE CONDITION
# ============================================================

print("\n" + "="*60)
print("SECTION 5 — POLITENESS FEATURES BY INCENTIVE CONDITION")
print("="*60)

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_politeness.csv")
    print(f"\n{model.upper()}:")
    print(df.groupby('condition')[KEY_POL].mean().round(3).to_string())

# ============================================================
# SECTION 6 — POLITENESS FEATURES — DARK PATTERN OUTPUTS ONLY BY CONDITION
# ============================================================

print("\n" + "="*60)
print("SECTION 6 — POLITENESS FEATURES — DARK PATTERN OUTPUTS ONLY BY CONDITION")
print("="*60)

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_politeness.csv")
    dark = df[(df['sycophancy']==1) | (df['anthropomorphisation']==1) | (df['user_retention']==1)]
    print(f"\n{model.upper()} — dark pattern outputs: {len(dark)}/{len(df)}")
    print(dark.groupby('condition')[KEY_POL].mean().round(3).to_string())

# ============================================================
# SECTION 7 — NRC-EIL EMOTION INTENSITY BY DARK PATTERN PRESENCE
# ============================================================

print("\n" + "="*60)
print("SECTION 7 — NRC-EIL EMOTION INTENSITY BY DARK PATTERN PRESENCE")
print("="*60)

EMOTIONS = ['anger_intensity', 'fear_intensity', 'joy_intensity', 'sadness_intensity']

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_nrceil.csv")
    print(f"\n{model.upper()}:")
    for label in LABELS:
        print(f"\n  {label}:")
        print(df.groupby(label)[EMOTIONS].mean().round(4).to_string())

# ============================================================
# SECTION 8 — NRC-EIL EMOTION INTENSITY BY INCENTIVE CONDITION
# ============================================================

print("\n" + "="*60)
print("SECTION 8 — NRC-EIL EMOTION INTENSITY BY INCENTIVE CONDITION")
print("="*60)

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_nrceil.csv")
    print(f"\n{model.upper()}:")
    print(df.groupby('condition')[EMOTIONS].mean().round(4).to_string())

# ============================================================
# SECTION 9 — STATISTICAL TESTS: POLITENESS (MANN-WHITNEY U)
# ============================================================

print("\n" + "="*60)
print("SECTION 9 — STATISTICAL TESTS: POLITENESS (MANN-WHITNEY U)")
print("="*60)

POL_FEATURES = ['Hedges', '1st_person', '2nd_person', 'Gratitude', 'total_politeness_strategies']

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_politeness.csv")
    print(f"\n{model.upper()}:")
    for label in LABELS:
        print(f"\n  {label}:")
        for feat in POL_FEATURES:
            p, sig, d = mannwhitney(df, label, feat)
            if p is not None:
                print(f"    {feat}: p={p:.4f} {sig}, d={d:.3f}")

# ============================================================
# SECTION 10 — STATISTICAL TESTS: NRC-EIL (MANN-WHITNEY U)
# ============================================================

print("\n" + "="*60)
print("SECTION 10 — STATISTICAL TESTS: NRC-EIL (MANN-WHITNEY U)")
print("="*60)

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_nrceil.csv")
    print(f"\n{model.upper()}:")
    for label in LABELS:
        print(f"\n  {label}:")
        for feat in EMOTIONS:
            p, sig, d = mannwhitney(df, label, feat)
            if p is not None:
                print(f"    {feat}: p={p:.4f} {sig}, d={d:.3f}")

# ============================================================
# SECTION 11 — JUDGE DISAGREEMENT SUMMARY
# ============================================================

print("\n" + "="*60)
print("SECTION 11 — JUDGE DISAGREEMENT SUMMARY")
print("="*60)

for model in MODELS:
    with open(f"{BASE}/{model}_final.json") as f:
        data = json.load(f)
    print(f"\n{model.upper()}:")
    for label in LABELS:
        j1_key = f"judge1_{label}"
        j2_key = f"judge2_{label}"
        j3_key = f"judge3_{label}"
        if j1_key not in data[0]:
            print(f"  {label}: judge votes not saved in output file")
            continue
        disagreements = sum(
            1 for d in data
            if len(set([d.get(j1_key,-1), d.get(j2_key,-1), d.get(j3_key,-1)]) - {-1}) > 1
        )
        print(f"  {label}: {disagreements}/{len(data)} disagreements ({100*disagreements/len(data):.1f}%)")

print("\n\nAnalysis complete.")

# ============================================================
# SECTION 12 — FULL 21-FEATURE POLITENESS ANALYSIS (SIGNIFICANT ONLY)
# ============================================================

print("\n" + "="*60)
print("SECTION 12 — FULL 21-FEATURE POLITENESS ANALYSIS (SIGNIFICANT ONLY)")
print("="*60)

ALL_POL_FEATURES = [
    'Please', 'Please_start', 'HASHEDGE', 'Indirect_(btw)', 'Hedges',
    'Factuality', 'Deference', 'Gratitude', 'Apologizing', '1st_person_pl.',
    '1st_person', '1st_person_start', '2nd_person', '2nd_person_start',
    'Indirect_(greeting)', 'Direct_question', 'Direct_start', 'HASPOSITIVE',
    'HASNEGATIVE', 'SUBJUNCTIVE', 'INDICATIVE'
]

for model in MODELS:
    df = pd.read_csv(f"{BASE}/{model}_politeness.csv")
    print(f"\n{model.upper()}:")
    for label in LABELS:
        print(f"\n  {label}:")
        g0 = df[df[label]==0]
        g1 = df[df[label]==1]
        found_any = False
        for feat in ALL_POL_FEATURES:
            if feat not in df.columns:
                continue
            p, sig, d = mannwhitney(df, label, feat)
            if p is not None and sig != 'ns':
                print(f"    {feat}: {sig}, d={d:.3f}")
                found_any = True
        if not found_any:
            print("    No significant features found")

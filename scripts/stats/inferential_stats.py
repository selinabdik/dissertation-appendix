import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy import stats

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

def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

# ============================================================
# SECTION 1: CHI-SQUARE BY CONDITION
# ============================================================
print("\n" + "="*60)
print("SECTION 1: DARK PATTERN RATES BY CONDITION (CHI-SQUARE)")
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
print("SECTION 2: DARK PATTERN RATES BY MODEL (CHI-SQUARE)")
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
print("SECTION 3: DARK PATTERN RATES BY PROMPT CATEGORY (CHI-SQUARE)")
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
# SECTION 4: KAPPA BY MODEL
# ============================================================
print("\n" + "="*60)
print("SECTION 4: COHEN'S KAPPA BY MODEL")
print("="*60)

df_val = pd.read_csv(f'{BASE}/validation_sample_150.csv')
df_val = df_val.dropna(subset=['human_sycophancy','human_anthropomorphisation','human_user_retention'])

print(f'Total annotated: {len(df_val)}')
for model in MODELS:
    subset = df_val[df_val['model'] == model]
    print(f'\n{model.upper()} (n={len(subset)}):')
    for label in LABELS:
        judge = subset[f'judge_{label}'].astype(int)
        human = subset[f'human_{label}'].astype(int)
        if len(set(judge)) > 1 and len(set(human)) > 1:
            kappa = cohen_kappa_score(human, judge)
            agreement = (judge == human).mean()
            print(f'  {label}: kappa={kappa:.3f}, agreement={agreement:.1%}')
        else:
            print(f'  {label}: insufficient variance for Kappa')

# ============================================================
# SECTION 5: McNEMAR'S TEST
# ============================================================
print("\n" + "="*60)
print("SECTION 5: McNEMAR'S TEST: JUDGE vs HUMAN")
print("="*60)

for label in LABELS:
    judge = df_val[f'judge_{label}'].astype(int).values
    human = df_val[f'human_{label}'].astype(int).values
    a = sum(1 for j,h in zip(judge,human) if j==0 and h==0)
    b = sum(1 for j,h in zip(judge,human) if j==0 and h==1)
    c = sum(1 for j,h in zip(judge,human) if j==1 and h==0)
    d = sum(1 for j,h in zip(judge,human) if j==1 and h==1)
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    print(f'\n{label}:')
    print(f'  False negatives (judge 0, human 1): {b}')
    print(f'  False positives (judge 1, human 0): {c}')
    print(f'  McNemar p={result.pvalue:.4f} {sig_stars(result.pvalue)}')
    if b > c:
        print(f'  Direction: judge systematically under-detects')
    elif c > b:
        print(f'  Direction: judge systematically over-detects')
    else:
        print(f'  Direction: errors are balanced')

# ============================================================
# SECTION 6: KRUSKAL-WALLIS BY CONDITION
# ============================================================
print("\n" + "="*60)
print("SECTION 6: KRUSKAL-WALLIS: EMOTION INTENSITY BY CONDITION")
print("="*60)

EMOTIONS = ['anger_intensity', 'fear_intensity', 'joy_intensity', 'sadness_intensity']

for model in MODELS:
    emo_df = pd.read_csv(f'{BASE}/{model}_nrceil.csv')
    print(f'\n{model.upper()}:')
    for feat in EMOTIONS:
        groups = [emo_df[emo_df['condition']==c][feat].values
                  for c in CONDITIONS]
        stat, p = kruskal(*groups)
        print(f'  {feat}: H={stat:.3f}, p={p:.4f} {sig_stars(p)}')

# ============================================================
# SECTION 7: BENJAMINI-HOCHBERG CORRECTION
# ============================================================
print("\n" + "="*60)
print("SECTION 7: BENJAMINI-HOCHBERG CORRECTION (MANN-WHITNEY)")
print("="*60)

POL_FEATURES = ['Hedges', '1st_person', '2nd_person', 'Gratitude',
                'total_politeness_strategies']

results = []
for model in MODELS:
    pol_df = pd.read_csv(f'{BASE}/{model}_politeness.csv')
    emo_df = pd.read_csv(f'{BASE}/{model}_nrceil.csv')
    for label in LABELS:
        for feat in POL_FEATURES:
            g0 = pol_df[pol_df[label]==0][feat].values
            g1 = pol_df[pol_df[label]==1][feat].values
            if len(g1) == 0: continue
            stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            d = cohens_d(g1, g0)
            results.append({'model': model, 'label': label,
                           'feature': feat, 'p': p, 'd': d})
        for feat in EMOTIONS:
            g0 = emo_df[emo_df[label]==0][feat].values
            g1 = emo_df[emo_df[label]==1][feat].values
            if len(g1) == 0: continue
            stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            d = cohens_d(g1, g0)
            results.append({'model': model, 'label': label,
                           'feature': feat, 'p': p, 'd': d})

results_df = pd.DataFrame(results)
reject, p_corrected, _, _ = multipletests(results_df['p'], method='fdr_bh')
results_df['p_corrected'] = p_corrected
results_df['sig_corrected'] = results_df['p_corrected'].apply(sig_stars)
results_df['sig_original'] = results_df['p'].apply(sig_stars)

print(f'Total tests: {len(results_df)}')
print(f'Significant before correction: {(results_df["p"] < 0.05).sum()}')
print(f'Significant after correction: {(results_df["p_corrected"] < 0.05).sum()}')
print('\nResults that changed significance after correction:')
changed = results_df[results_df['sig_original'] != results_df['sig_corrected']]
if len(changed) == 0:
    print('  None - all results maintained significance')
else:
    print(changed[['model','label','feature','p','p_corrected','sig_original','sig_corrected','d']].to_string())

print("\nDone.")

# 10-Round Analysis Protocol for Wearable Biomarker Discovery

## Round 1: Data Cleaning & Feature Engineering

```python
import pandas as pd
import numpy as np

# Load and inspect
df = pd.read_csv('data.csv')
print(f"Raw N={len(df)}, columns={len(df.columns)}")

# Check outcome prevalence
outcome_col = 'phq9_total'  # adapt to your dataset
threshold = 10
prev = (df[outcome_col] >= threshold).mean()
print(f"Prevalence: {prev:.1%} ({(df[outcome_col] >= threshold).sum()} cases)")
# Rule of thumb: need ≥100 cases for stable logistic regression

# Drop subjects with >50% missing wearable features
feat_cols = [c for c in df.columns if c.startswith(('rhr_', 'sleep_', 'steps_', 'hrv_'))]
df = df.dropna(subset=[outcome_col])
missing_rate = df[feat_cols].isnull().mean(axis=1)
df = df[missing_rate < 0.5]
print(f"Clean N={len(df)}")

# Save clean data and feature list
df.to_csv('subjects_clean.csv', index=False)
import json
json.dump(feat_cols, open('feat_cols.json', 'w'))
```

**Key checks:**
- Document N at each filtering step (for CONSORT flow)
- Verify outcome is not constant (SD > 0)
- Check if outcome is single-timepoint vs longitudinal (ICC analysis)
- Log feature count and types

## Round 2: Feature Screening

```python
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

binary = (df[outcome_col] >= threshold).astype(int)
results = []
for feat in feat_cols:
    valid = df[feat].notna()
    if valid.sum() < 100:
        continue
    rho, p = spearmanr(df.loc[valid, feat], binary[valid])
    results.append({'feature': feat, 'rho': rho, 'p': p, 'n': valid.sum()})

res = pd.DataFrame(results)
_, res['p_fdr'], _, _ = multipletests(res['p'], method='fdr_bh')
sig = res[res['p_fdr'] < 0.05].sort_values('rho', key=abs, ascending=False)
print(f"FDR-significant: {len(sig)}/{len(res)} features")
```

**Expect:** 30-50% of features will be FDR-significant in large samples. This does NOT mean they're useful — effect sizes matter more than p-values.

## Round 3: Effect Sizes & Redundancy

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# OR per SD for each feature
scaler = StandardScaler()
for feat in sig['feature']:
    X = scaler.fit_transform(df[[feat]].dropna())
    y = binary[df[feat].notna()]
    lr = LogisticRegression().fit(X, y)
    or_per_sd = np.exp(lr.coef_[0][0])
    print(f"{feat}: OR={or_per_sd:.3f}")

# Redundancy check — correlation matrix of top features
top_feats = sig.head(20)['feature'].tolist()
corr = df[top_feats].corr(method='spearman').abs()
# Features with r > 0.85 are redundant — keep the more interpretable one
```

**Key insight:** After redundancy pruning, you'll typically have 2-5 truly independent signals. This is normal.

## Round 4: Validation Split

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_disc, X_val, y_disc, y_val = train_test_split(
    df[independent_feats], binary, test_size=0.4, stratify=binary, random_state=42
)

# Fit on discovery
lr = LogisticRegression().fit(StandardScaler().fit_transform(X_disc), y_disc)
# Evaluate on validation
auc_val = roc_auc_score(y_val, lr.predict_proba(StandardScaler().fit_transform(X_val))[:, 1])
print(f"Validation AUC: {auc_val:.3f}")

# Feature replication: which discovery features remain significant in validation?
for feat in independent_feats:
    rho_val, p_val = spearmanr(X_val[feat], y_val)
    print(f"{feat}: val_rho={rho_val:.3f}, val_p={p_val:.4f}")
```

**Benchmark:** 50-75% feature replication is good. AUC drop of 0.02-0.05 from discovery to validation is normal.

## Round 5: Symbolic Regression & Advanced Discovery

```python
# Only attempt if you have strong signal (AUC > 0.65 with simple features)
# SR typically fails in weak-signal wearable regimes
from gplearn.genetic import SymbolicRegressor
sr = SymbolicRegressor(
    population_size=3000, generations=50,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs'],
    n_jobs=1,  # MUST be 1 — multiprocessing breaks
    random_state=42
)
# If R² < 0 on test set, SR has failed. Move on. Don't tune endlessly.
```

## Round 6: Non-Linearity & Dose-Response

```python
# Quartile analysis for dose-response
for feat in independent_feats:
    df['quartile'] = pd.qcut(df[feat], 4, labels=['Q1','Q2','Q3','Q4'])
    for q in ['Q1','Q2','Q3','Q4']:
        prev_q = binary[df['quartile'] == q].mean()
        rr = prev_q / binary[df['quartile'] == 'Q1'].mean()
        print(f"{feat} {q}: prevalence={prev_q:.3f}, RR={rr:.2f}")

# Test non-linearity with polynomial term
from sklearn.preprocessing import PolynomialFeatures
# If quadratic term is significant (p < 0.05), the relationship is non-linear
```

**Expect:** RHR and sleep variability often show non-linear dose-response (risk accelerates in top quartile).

## Round 7: Cross-System Interactions

```python
# Test products and ratios across physiological systems
# e.g., cardiac × sleep, cardiac × activity
interaction_pairs = [
    ('rhr_mean', 'sleep_cv'),      # cardiac × sleep
    ('rhr_mean', 'cardio_cv'),     # cardiac × fitness
    ('sleep_cv', 'steps_mean'),    # sleep × activity
]
for f1, f2 in interaction_pairs:
    df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    rho, p = spearmanr(df[f'{f1}_x_{f2}'].dropna(), binary[df[f'{f1}_x_{f2}'].notna()])
    # Compare to individual features — interaction should exceed both components
```

**This is where novelty lives.** Cross-system interactions are understudied and can exceed individual feature associations.

## Round 8: Representation Learning

```python
from sklearn.decomposition import PCA

# PCA baseline
pca = PCA(n_components=8)
X_pca = pca.fit_transform(StandardScaler().fit_transform(df[feat_cols].fillna(df[feat_cols].median())))
auc_pca = # ... logistic regression on PCA components

# Autoencoder (only if you have >1000 subjects)
# Typically: PCA matches or beats autoencoder for wearable data
# Report honestly if autoencoder doesn't help
```

## Round 9: Clinical Utility & Confounding

```python
# PPV at realistic operating points
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
# PPV = precision at the threshold where you'd actually deploy

# E-values for unmeasured confounding robustness
# E-value = OR + sqrt(OR * (OR - 1))
for feat, or_val in zip(independent_feats, adjusted_ors):
    e_value = or_val + np.sqrt(or_val * (or_val - 1))
    print(f"{feat}: OR={or_val:.3f}, E-value={e_value:.2f}")
# E-values < 2.0 mean modest confounding could explain the association
```

## Round 10: Synthesis

Compile all results into an honest assessment:
- How many features survived all filters?
- What's the ceiling AUC? Does it beat demographics alone?
- Did any advanced method beat logistic regression?
- What are the 2-3 genuinely actionable findings?
- What should future work focus on?

Write results to a structured output directory for paper generation.

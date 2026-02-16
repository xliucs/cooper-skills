---
name: wearable-biomarker-discovery
description: Systematic biomarker discovery from wearable sensor data for mental health outcomes (depression, anxiety). Covers the full pipeline from raw daily wearable data to NeurIPS-quality paper: subject-level aggregation, 273+ feature engineering (summary stats, temporal dynamics, cross-system interactions, frequency-domain, information-theoretic, automated extraction), confounder adjustment, bootstrap validation, dose-response analysis, latent variable methods (FA, CCA, autoencoders), and honest null reporting. Designed for longitudinal wearable datasets with psychiatric assessments (PHQ-9, GAD-7).
---

# Wearable Biomarker Discovery

Systematic protocol for discovering digital biomarkers of mental health conditions from consumer wearable data. Built from 10 rounds of iterative analysis on the DWB dataset (N=4,451, Fitbit, PHQ-9/GAD-7) with 4 rounds of expert reviewer feedback.

## First: Read PROJECT_STATE.md

If `PROJECT_STATE.md` exists, read it FIRST. It contains progress, results, and what's been tried.

## The Core Principle

**Iterate on ANALYSIS, not on WRITING.** Each round must produce new code, new experiments, and new findings. Paper writing comes AFTER all analysis rounds are complete. "Iterating 10 times" means 10 rounds of genuine analysis, not 10 rounds of editing the same paper.

## Phase 1: Data Foundation (Round 1)

### 1.1 Raw Data Inspection
```python
# Load and inspect
raw = pd.read_csv('data.csv')
print(f"Rows: {raw.shape[0]:,}, Subjects: {raw['user_id'].nunique():,}")
print(f"Columns: {raw.columns.tolist()}")
```

### 1.2 Subject-Level Aggregation
**CRITICAL**: Most wearable datasets have repeated daily measures per subject with static psychiatric outcomes (collected once). You MUST aggregate to one row per subject before any analysis.

```python
# Check ICC first — is the outcome static or dynamic?
per_user_std = raw.groupby('user_id')['outcome'].std().mean()
# If std ≈ 0: outcome is static, use between-person analysis only
# If std > 0: within-person analysis is possible (rare for PHQ-9)
```

**PHQ-9 NaN extraction bug**: Do NOT use `groupby().first()` — it picks the first row which may have NaN. Use:
```python
phq_per_user = raw.groupby('user_id')['phq_score'].apply(
    lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
)
```

### 1.3 Feature Engineering (6 Categories, ~273 features)

**A. Summary statistics** (~150 features): Per-signal mean, SD, CV, median, IQR, skewness, min, max, range for each wearable signal.

**B. Temporal dynamics** (~50 features): Daily change magnitude (abs diff mean), autocorrelation, linear trend (slope via linregress), weekend-weekday differentials.

**C. Cross-system interactions** (~15 features): 
- Sympathovagal ratio: RHR/RMSSD
- Cardiac strain: RHR_SD × RHR_mean
- Activity-recovery mismatch: steps/RMSSD
- Pairwise daily correlations: sleep-RHR, sleep-steps, HRV-steps
- Multi-system coherence: mean |r| across all pairs
- Cardiac recovery efficiency: RMSSD per hour of sleep
- Activity intensity ratios: peak/(peak+cardio+fat_burn)

**D. Interaction terms** (~20 features): Pairwise products and ratios of top 5 features. Look for SYNERGISTIC interactions where |ρ_interaction| > max(|ρ_component1|, |ρ_component2|).

**E. Information-theoretic** (2+ features): Permutation entropy (order 3) for key time series.

**F. Automated extraction**: 
- tsfresh with MinimalFCParameters (~40 features, but ~65% redundant with manual features)
- Symbolic regression via gplearn (expect null results — report them honestly)
- Autoencoder bottleneck representations

### 1.4 Data Leakage Prevention
**CRITICAL**: Check for derived outcome columns in features. Common leakage sources:
- `dep_mod`, `dep_mild`, `dep_severe` — binary indicators from PHQ
- PHQ item scores used as predictors
- Any column with perfect or near-perfect correlation with the outcome

```python
# Strict exclusion
leakage_patterns = ['phq', 'gad', 'dep_', 'depression_score', 'anxiety_score']
feat_cols = [c for c in df.columns if not any(p in c.lower() for p in leakage_patterns)]
```

## Phase 2: Statistical Screening (Round 2)

### 2.1 Global FDR Screening
```python
from statsmodels.stats.multitest import multipletests
results = []
for f in feat_cols:
    rho, p = stats.spearmanr(df[f], df['phq_score'])
    results.append({'feature': f, 'rho': rho, 'p': p})
screen = pd.DataFrame(results)
_, fdr_p, _, _ = multipletests(screen['p'], method='fdr_bh')
screen['p_fdr'] = fdr_p
```

### 2.2 Confounder Adjustment
Always adjust for age, gender, SES. Report attenuation percentage.
```python
# Partial Spearman via residualization
from sklearn.linear_model import LinearRegression
confounders = ['age', 'gender_score', 'financial_situation_score']
# Residualize both feature and outcome on confounders, then correlate residuals
```

### 2.3 E-values for Unmeasured Confounding
```python
def e_value(or_val):
    if or_val < 1: or_val = 1/or_val
    return or_val + np.sqrt(or_val * (or_val - 1))
```
Interpretation: An unmeasured confounder must be associated with both biomarker and outcome at RR ≥ E-value to explain away the finding.

## Phase 3: Advanced Analyses (Rounds 3-8)

### 3.1 Symbolic Regression (expect null results)
```python
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
# Use pop≥2000, gen≥50, multiple seeds
# Report ALL results including degenerate expressions
# If R² < 0: this IS the finding — no non-linear signal detectable
```

### 3.2 Dose-Response Analysis
```python
# Quintile analysis with F-test for non-linearity
# Test quadratic term: if p < 0.05, relationship is non-linear
c2 = np.polyfit(x, y, 2)  # quadratic
c1 = np.polyfit(x, y, 1)  # linear
f_stat = ((ss_linear - ss_quadratic) / 1) / (ss_quadratic / (n - 3))
```

### 3.3 Depression vs Anxiety Comparison (Steiger's Test)
```python
def steiger_test(r12, r13, r23, n):
    """Compare r(biomarker,depression) vs r(biomarker,anxiety)"""
    r_bar = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - r_bar**2))
    h = (1 - f * r_bar**2) / (1 - r_bar**2)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    z_diff = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - norm.cdf(abs(z_diff)))
    return z_diff, p
```
**CAVEAT**: In mostly-healthy samples, PHQ-GAD correlation is deflated by range restriction (r≈0.25 vs r≈0.50-0.70 in clinical samples). This inflates Steiger test power. Always note this.

### 3.4 Bootstrap Steiger (500+ iterations)
For each bootstrap sample, compute both correlations and their difference. Report CI of difference and % of iterations where anxiety > depression.

### 3.5 Factor Analysis
Run TWICE: once on mean-level features, once on variability features.
- Mean FA may miss variability-driven signal (sleep factors may be non-significant for PHQ despite individual sleep variability features being significant)
- Variability FA often reveals cardiac and respiratory dimensions missed by mean FA

### 3.6 CCA (Canonical Correlation Analysis)
Use joint means + variability features vs PHQ-9 items. Inspect which PHQ items load highest — typically fatigue/sleep dominate (wearables capture somatic, not cognitive/affective symptoms).

### 3.7 Autoencoder Latent Biomarkers
Bottleneck architecture (e.g., 84→32→8→32→84). Compare against PCA with same dimensionality. In our experience, PCA outperforms autoencoders for this task — non-linear latent representations don't help.

### 3.8 Cross-System Interaction Search
Test all pairwise products AND ratios of top features. Flag "synergistic" interactions where the combined signal exceeds either component alone.

## Phase 4: Validation (Round 9)

### 4.1 Internal Validation (70/30 subject split)
```python
np.random.seed(42)
users = df['user_id'].unique()
np.random.shuffle(users)
split = int(0.7 * len(users))
disc = df[df['user_id'].isin(users[:split])]
val = df[df['user_id'].isin(users[split:])]
```
- Screen features in discovery (FDR)
- Replicate in validation (same direction, p<0.05)
- Train model on discovery, evaluate on validation

### 4.2 Information Ceiling
```python
# k-NN provides non-parametric ceiling; LR provides practical baseline
# MUST use cross-validation, NOT resubstitution
# If k-NN ≤ LR: no non-linear signal exploitable
```

## Phase 5: Integration (Round 10)

### 5.1 Model Comparison / Ablation
Always compare:
1. Demographics only (age, gender, SES)
2. Wearables only
3. Combined
4. Minimal model (top 2 features only)

Report wearable INCREMENT over demographics — this is the honest contribution.

### 5.2 Clinical Utility
Compute sensitivity, specificity, PPV at operating points. At low prevalence (e.g., 5% for PHQ≥10), PPV will be very low (~5-7%) regardless of AUC. State this explicitly.

## Phase 6: Paper Writing

Write ONLY after all analysis rounds complete. Structure:
1. Frame confirmatory findings honestly — discovering known biomarkers IS a contribution at large scale
2. Highlight genuine novel discoveries (synergistic interactions, anxiety finding, non-linearity)
3. Report ALL null results (SR failure, AE non-superiority, tsfresh redundancy)
4. Include comment-response table if revising

## Key Domain Knowledge

### Expected Effect Sizes (Jacobson et al. 2019)
- Wearable → depression: r = 0.10–0.30, R² = 0.05–0.15
- Anything above r = 0.40 should be treated with suspicion (likely confound or leakage)
- Large-sample AUCs (N>1000): 0.55–0.65 for single-modality wearables

### Known Biomarkers (DO NOT claim as novel)
- Reduced HRV (RMSSD) → depression (Koch 2019, Kemp 2010)
- Reduced sleep duration → depression (Baglioni 2016)
- Altered REM% → depression (Baglioni 2016)
- Reduced physical activity → depression (Schuch 2018, Pearce 2022)
- Poor sleep efficiency → depression (Scott 2021)
- Elevated RHR → depression (Kemp 2010)
- Later sleep midpoint → depression (Walch 2016)

### What IS Genuinely Novel (examples from our work)
- Sleep VARIABILITY (SD, CV) outperforming sleep MEAN
- Cross-system synergistic interactions (RHR × cardio CV)
- Anxiety > depression signal (formalized with Steiger's test)
- Non-linear dose-response with inflection points
- Wearable signals detecting subclinical distress, not clinical severity

### Common Pitfalls
1. **Data leakage from derived outcome columns** (dep_mod, dep_mild, etc.)
2. **PHQ NaN extraction with .first()** — use .apply(lambda x: x.dropna().iloc[0])
3. **k-NN ceiling on training data = 1.0** — always use CV
4. **Reporting sleep SD and CV as separate discoveries** — CV = SD/mean, they're r=0.93
5. **tsfresh mostly redundant** — 65% overlap with basic summary stats
6. **Small-sample AUC inflation** — AUC 0.89 at N=31 vs 0.60 at N=4,451 is expected, not a failure
7. **Demographic confounding** — demographics alone can match wearable models (AUC 0.64-0.68)
8. **Range-restricted PHQ-GAD correlation** — deflated in healthy samples, inflates Steiger power
9. **Frequency-domain features unreliable at <4 weeks** — need ≥3 weekly cycles for spectral estimates
10. **Severity paradox**: biomarker signal concentrates in subclinical range (PHQ 0-4), disappears at clinical thresholds

## Scripts

### Core analysis scripts (in `scripts/` directory)
- `plot_style.py` — Consistent figure styling
- `error_analysis.py` — Model error diagnosis
- `ceiling_analysis.py` — k-NN ceiling estimation

## References

See `references/` for:
- `domain_knowledge.md` — Comprehensive literature on wearable biomarkers for depression
- `statistical_methods.md` — Details on Steiger's test, E-values, bootstrap methods

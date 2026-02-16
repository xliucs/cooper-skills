# Statistical Tests for Wearable Biomarker Studies

## Steiger's Test (Comparing Dependent Correlations)

Use when comparing whether a biomarker associates more strongly with outcome A vs outcome B (e.g., depression vs anxiety).

```python
from scipy.stats import norm
import numpy as np

def steiger_test(r_xz, r_yz, r_xy, n):
    """Test if r(X,Z) differs from r(Y,Z) where X,Y share subjects."""
    r_det = 1 - r_xz**2 - r_yz**2 - r_xy**2 + 2*r_xz*r_yz*r_xy
    r_bar = (r_xz + r_yz) / 2
    denom = np.sqrt(
        (2 * (1 - r_xy) * (1 - r_bar**2)**2) /
        (n * (1 - r_det / (1 - r_bar**2)))
    )
    z = (r_xz - r_yz) / denom if denom > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p
```

Bootstrap it (500+ iterations) for robustness. Report consistency percentage.

## E-Values (Sensitivity to Unmeasured Confounding)

```python
def e_value(or_val):
    """Minimum confounding strength to explain away association."""
    return or_val + np.sqrt(or_val * (or_val - 1))
```

- E-value < 1.5: very fragile to confounding
- E-value 1.5-2.0: modest robustness
- E-value > 2.0: reasonably robust

For wearable biomarkers with OR ~1.2, expect E-values ~1.6. Be honest about this.

## Bootstrap Confidence Intervals

```python
def bootstrap_ci(data, stat_func, n_boot=1000, ci=0.95):
    stats = []
    for _ in range(n_boot):
        sample = data.sample(n=len(data), replace=True)
        stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    return np.percentile(stats, [alpha*100, (1-alpha)*100])
```

Always report 95% CIs for ORs and AUCs. If CI crosses 1.0 (for OR) or 0.5 (for AUC), the finding is not robust.

## Canonical Correlation Analysis (CCA)

Use to find multivariate associations between wearable feature sets and symptom sets.

```python
from sklearn.cross_decomposition import CCA as SkCCA

cca = SkCCA(n_components=3)
cca.fit(X_wearable_scaled, Y_symptoms_scaled)
X_c, Y_c = cca.transform(X_wearable_scaled, Y_symptoms_scaled)
# Canonical correlations
for i in range(3):
    cc = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    print(f"CC{i+1}: {cc:.3f}")
```

Typical CC1 for wearable-mental-health: 0.20-0.30. Report loadings to interpret what each variate captures.

## Factor Analysis

Use to identify latent wearable "systems" (cardiac, sleep, activity).

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=5, rotation='varimax')
fa.fit(X_scaled)
loadings = pd.DataFrame(fa.components_.T, index=feat_cols, columns=[f'F{i+1}' for i in range(5)])
```

Run separately on: (1) mean features, (2) variability features. They capture different physiology.

## PHQ-GAD Range Restriction

If PHQ-9 and GAD-7 correlate lower than expected (~0.25 vs literature's 0.60+), check for range restriction. Wellness populations truncate the clinical range, deflating correlations. Report corrected r using Thorndike's formula or note the limitation.

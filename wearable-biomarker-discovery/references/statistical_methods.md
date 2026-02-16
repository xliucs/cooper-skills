# Statistical Methods Reference

## Steiger's Test for Comparing Dependent Correlations

Tests whether r(X,Y1) differs from r(X,Y2) when Y1 and Y2 are correlated.

```python
from scipy.stats import norm

def steiger_test(r12, r13, r23, n):
    """
    r12: correlation between biomarker and outcome 1 (e.g., depression)
    r13: correlation between biomarker and outcome 2 (e.g., anxiety)
    r23: correlation between outcome 1 and outcome 2
    n: sample size
    """
    r_bar = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - r_bar**2))
    h = (1 - f * r_bar**2) / (1 - r_bar**2)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    z_diff = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - norm.cdf(abs(z_diff)))
    return z_diff, p
```

**Caveat**: In range-restricted samples (mostly healthy), the Y1-Y2 correlation is deflated, inflating test power. Always report the Y1-Y2 correlation and compare to expected clinical values.

## E-values for Unmeasured Confounding

```python
def e_value(or_val):
    """Minimum confounding RR to explain away observed OR"""
    if or_val < 1:
        or_val = 1/or_val
    return or_val + np.sqrt(or_val * (or_val - 1))
```

Interpretation: If E-value = 1.83, an unmeasured confounder must be associated with both the biomarker and outcome at RR ≥ 1.83 to fully explain the observed association.

## Bootstrap Steiger's Test

```python
n_boot = 500
diffs = []
for b in range(n_boot):
    idx = np.random.choice(len(df), len(df), replace=True)
    boot = df.iloc[idx]
    r_dep = stats.pearsonr(boot[feat], boot['phq_score'])[0]
    r_anx = stats.pearsonr(boot[feat], boot['gad_score'])[0]
    diffs.append(abs(r_anx) - abs(r_dep))
ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
```

## Post-Hoc Power Analysis

```python
from scipy.stats import norm

def power_for_or(or_val, n_cases, n_total, alpha=0.05):
    prevalence = n_cases / n_total
    se = 1 / np.sqrt(n_total * prevalence * (1 - prevalence))
    z = np.log(or_val) / se
    z_crit = norm.ppf(1 - alpha/2)
    power = 1 - norm.cdf(z_crit - z) + norm.cdf(-z_crit - z)
    return power
```

## Non-linearity F-test

```python
# Compare linear vs quadratic fit
c1 = np.polyfit(x, y, 1)
c2 = np.polyfit(x, y, 2)
ss_lin = np.sum((y - np.polyval(c1, x))**2)
ss_quad = np.sum((y - np.polyval(c2, x))**2)
f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
p_nonlin = 1 - scipy.stats.f.cdf(f_stat, 1, n - 3)
```

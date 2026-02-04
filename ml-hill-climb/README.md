# ml-hill-climb

An OpenClaw skill for autonomous ML/data science hill-climbing on tabular data. Designed for long-horizon tasks (4-24 hours) with minimal human intervention.

## Why This Skill Exists

This skill encodes hard-won lessons from multiple ML projects where an AI agent required heavy human intervention to stay productive. The core problems it solves:

| Problem | What Happens Without the Skill | How the Skill Fixes It |
|---------|-------------------------------|----------------------|
| Random model shopping | Agent tries 20+ architectures without analyzing why previous ones failed | Every experiment requires a stated hypothesis with evidence |
| No ceiling analysis | Agent grinds for hours toward an impossible target | Phase 1 mandates theoretical ceiling estimation before modeling |
| Silent grinding | Agent runs for 2-3 hours with no updates, might be stuck | 30-min circuit breaker: must update or flag if stuck |
| Inflated results | Agent reports leaked/overfit scores without sanity checks | Quality gate: verify any >5% jump, check train-test gap |
| Analysis only at the end | Best insights come from error analysis, but it's done last | Analysis checkpoints every 3-5 versions drive next hypothesis |
| Catastrophic forgetting | Agent loses track of what was tried across context windows | PROJECT_STATE.md as persistent structured memory |
| Poor figure quality | First-pass figures have overlapping legends, inconsistent style | Shared plot_style.py, publication-quality from version 1 |

## The Protocol

```
Phase 1: Understand (20%)     →  EDA, baselines, ceiling analysis, subgroup analysis
Phase 2: Experiment (50%)     →  Hypothesis-driven: analyze → hypothesize → experiment → analyze
Phase 3: Write Up (30%)       →  Post-hoc analysis, publication figures, paper/report
```

The key insight: **analysis drives experiments in ALL phases**, not just at the end. The cycle is always: analyze → hypothesize → experiment → analyze.

## Example Prompts

### Regression Task

```
Dataset: /path/to/data.csv (1000 samples, 25 features)
Task: Predict "target_column" (continuous). Metric: R².
Reference: prior work achieves R²=0.50.
Eval: 5-fold × 5-repeat stratified CV, seed 42. Fixed eval framework for ALL experiments.
Report: post every version to [Telegram group / Slack channel / etc.]
GitHub: push to [repo], update README after each version.
Time budget: 8 hours.

Start with EDA and ceiling analysis BEFORE modeling.
Every experiment needs a hypothesis.
Error analysis every 3-5 versions.
If 3 versions show <0.002 gain and analysis confirms signal exhaustion, write up with publication figures.
If stuck 30 min, message me.
```

### Classification Task

```
Dataset: /path/to/data.csv (5000 samples, 40 features)
Task: Predict "label" (binary classification). Metric: AUC-ROC (also report F1, precision, recall).
Class balance: ~20% positive.
Reference: logistic regression baseline AUC=0.75.
Eval: 5-fold stratified CV, seed 42.
Report: post every version to [channel].
Time budget: 6 hours.

Start with EDA and class balance analysis.
Try class weighting and threshold tuning early.
Error analysis: confusion matrix patterns, which subgroups are misclassified.
Write up when converged.
```

### Multi-Target Task

```
Dataset: /path/to/data.csv
Task: Predict both "target_A" (R²) and "target_B" (R²).
Two model variants each:
  - Model A: all features
  - Model B: subset of features only (no columns X, Y, Z)
Reference: paper achieves A=0.50, B=0.21.
Eval: shared CV splits across all models.
Report: [channel]. Time budget: 12 hours.

Prioritize target_A first (more signal expected), then target_B.
Ceiling analysis for both targets.
```

### What Makes These Prompts Work

Compared to a naive prompt like *"here's a dataset, predict X, target R²=0.85"*:

1. **Specifies eval protocol** — no inconsistent CV splits across versions
2. **Sets time budget** — prevents infinite grinding
3. **Mandates ceiling analysis first** — agent knows when to stop
4. **Requires hypotheses** — prevents random model shopping
5. **Includes reporting cadence** — no silent hours
6. **Has exit criteria** — convergence check + signal exhaustion analysis
7. **References prior work** — realistic expectations, not arbitrary targets

## Contents

```
ml-hill-climb/
├── SKILL.md                          # Protocol (auto-loaded by OpenClaw)
├── README.md                         # This file
├── scripts/
│   ├── ceiling_analysis.py           # k-NN neighbor variance R² ceiling estimator
│   ├── error_analysis.py             # Residual correlations, subgroups, error correlation
│   └── plot_style.py                 # Blue-palette publication plotting style
└── references/
    └── checklist.md                  # Phase-by-phase checklist
```

## Scripts

### ceiling_analysis.py

Estimates theoretical maximum R² for a regression task using k-NN neighbor variance analysis. For each sample, finds k nearest neighbors in feature space and measures target variance among them — this variance is irreducible noise that no model can overcome.

```bash
python scripts/ceiling_analysis.py --data data.csv --target y --k 10
# Also runs sensitivity analysis for k=5,10,15,20,30
```

### error_analysis.py

Comprehensive error analysis toolkit:
- **Residual-feature correlations**: are there features with unused predictive signal?
- **Subgroup analysis**: performance breakdown by any categorical grouping
- **Error correlation matrix**: do different models fail on the same samples? (>0.95 = no ensemble benefit)
- **Target range analysis**: where in the target distribution does the model fail?

### plot_style.py

Shared style for publication-quality figures. Blue-dominant palette, clean spines, consistent font sizes. Import and call `setup_style()` at the top of any plotting script.

## Installation

Copy to your OpenClaw workspace skills directory:

```bash
cp -r ml-hill-climb /path/to/clawd/skills/
```

The skill auto-triggers whenever the agent receives an ML/data science modeling task.

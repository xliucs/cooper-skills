---
name: ml-hill-climb
description: Autonomous ML/data science hill-climbing for tabular regression and classification tasks. Use when asked to build predictive models, optimize R²/accuracy on a dataset, run ML experiments, or do any iterative model improvement project. Covers the full lifecycle from EDA through modeling to paper-quality writeup.
---

# ML Hill-Climb

Autonomous protocol for iterative ML model improvement on tabular data. Designed for long-horizon tasks (4-24 hours) with minimal human intervention.

## First: Read PROJECT_STATE.md

If `PROJECT_STATE.md` exists in the working directory, read it FIRST. It contains your current progress, best results, active hypotheses, and what's been tried. **This is your memory across sessions.**

If it doesn't exist, create it after Phase 1.

## The Core Loop

Every experiment follows: **analyze → hypothesize → experiment → analyze**

Never run an experiment without articulating WHY it should help. If you can't state "I expect +X improvement because analysis shows Y" — stop and analyze more.

## Phase 1: Understand (first 20% of time budget)

Before touching a model:

1. **EDA**: distributions, correlations, missingness, outliers, class balance (for classification)
2. **Eval framework**: create `eval_framework.py` with FIXED CV splits (stratified, seed=42). Use for ALL experiments. Never change it.
3. **Task-appropriate baselines**:
   - Regression: ElasticNet + XGBoost
   - Classification: LogisticRegression + XGBoost
   - These set the floor.
4. **Ceiling analysis**: run `scripts/ceiling_analysis.py` (regression) to estimate theoretical max. Also do domain reasoning — is there a formula relating target to features? What information is missing? What does prior work achieve?
5. **Subgroup analysis**: performance by key categorical splits and feature quantiles
6. **Feature space visualization**: t-SNE or PCA colored by target, prediction error, key features
7. **Output**: post summary to reporting channel with sample count, feature count, target stats, ceiling estimate, baseline score

Create `PROJECT_STATE.md` with all findings.

## Phase 2: Hypothesis-Driven Experiments (next 50%)

### Every experiment MUST have:
- **Hypothesis**: "I expect +X because [analysis finding]"
- **Method**: what you're trying
- **Result**: what happened
- **Analysis**: WHY did it work/not work?

### After every version:
- `git commit` with results
- Update `README.md` results table
- Update `PROJECT_STATE.md`
- Post to reporting channel: version / score / what / why / next

### Analysis checkpoints (every 3-5 versions):
- Residual-feature correlations (regression) or confusion patterns (classification) → unused signal?
- Subgroup breakdown → which groups improved/worsened?
- Error correlation across models → ensemble diversity?
- t-SNE of errors → failures clustered or dispersed?
- Use findings to generate NEXT hypothesis

### Typical impact order (regression):
1. Target transforms (log1p for right-skewed, sqrt for moderate skew)
2. Feature engineering (domain interactions, ratios, polynomial terms)
3. Sample weighting (upweight underrepresented regions of target)
4. Hyperparameter tuning (Optuna, 50-100 trials)
5. Diverse model blending (trees + linear for different error patterns)
6. Input transforms (QuantileTransformer, StandardScaler)
7. Loss functions (MAE/Huber for outlier robustness)

### Typical impact order (classification):
1. Class imbalance handling (SMOTE, class weights, threshold tuning)
2. Feature engineering (domain interactions, ratios)
3. Hyperparameter tuning (Optuna, 50-100 trials)
4. Calibration (Platt scaling, isotonic)
5. Diverse model blending (trees + linear + SVM)
6. Threshold optimization (for F1, precision-recall tradeoff)

### Common pitfalls (verify with analysis before trying):
- Blending more of the same model family when error correlations >0.95
- Stacking without nested CV (leakage risk — always verify)
- Deep learning at n<2000 on tabular data (trees almost always win)
- Aggressive feature selection (trees handle irrelevant features naturally)
- Post-hoc calibration/stretching (rarely helps, can hurt)

## Phase 3: Write Up (last 30%)

Enter Phase 3 when:
- 3 consecutive versions with <0.5% relative improvement AND analysis confirms signal exhausted
- Gap to theoretical ceiling < estimation error
- Time budget running out

Produce:
1. Comprehensive error analysis with publication-quality figures
2. Technical report or paper draft:
   - Experiment journey (what was tried, what worked, what didn't)
   - Why the ceiling exists (information-theoretic or domain analysis)
   - Key findings (feature importance, failure modes, subgroup disparities)
3. Figures: consistent style (use `scripts/plot_style.py`), squared panels, clean legends

## Communication Rules

- **Never go >30 min without posting an update** (even "running fold 3/5")
- **If stuck 30 min with no improvement**: message — "stuck at X, tried Y, hypothesis is Z"
- **If 3 experiments show negligible gain**: message before trying a 4th
- **If uncertain whether approach is sound**: ask before spending an hour on it

## Quality Gates

- **Leakage check**: if score jumps >5% relative in one version, verify before reporting. Compare train vs test scores.
- **Overfitting vs ceiling**: if train-test gap is large, diagnose — try shallower models, more regularization, or accept information ceiling
- **Reproducibility**: fixed random seeds, deterministic CV splits, reproducible results
- **Figures**: publication-quality from version 1. Use `scripts/plot_style.py`.

## PROJECT_STATE.md Template

Maintain in working directory. Update every version.

```markdown
# Project State

## Task
- Target: [variable name]
- Metric: [R², accuracy, F1, AUC, etc.]
- Data: [path, N samples, M features]
- Report to: [channel/method]
- Time budget: [hours remaining]

## Current Best
- [Score and method for each model variant]

## Theoretical Ceiling
- [Estimated max score, method used, key assumptions]

## Versions Tried
| V | Method | Score | Key Finding |
|---|--------|-------|-------------|

## Active Hypotheses
1. [Hypothesis + expected gain + supporting evidence]

## Analysis Summary
- Residual-feature correlations: [near zero = exhausted, or which features still correlate]
- Error correlation across models: [>0.95 = no diversity, <0.90 = diversity exists]
- Subgroup performance: [which groups are easy/hard, any disparities]
- Signal exhaustion: [yes/no + evidence]

## Next Steps
1. [Prioritized by expected impact]
```

## Scripts

- `scripts/ceiling_analysis.py` — k-NN neighbor variance ceiling estimator (regression)
- `scripts/error_analysis.py` — residual correlations, subgroup breakdown, model error correlation
- `scripts/plot_style.py` — shared blue-palette publication plotting style

See `references/checklist.md` for phase-by-phase checklist.

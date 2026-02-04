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

1. **EDA**: distributions, correlations, missingness, outliers, class balance
2. **Eval framework**: create `eval_framework.py` with FIXED CV splits (stratified, seed=42). Use this for ALL experiments. Never change it.
3. **Baselines**: run ElasticNet + one tree model (XGBoost). These set the floor.
4. **Ceiling analysis**: run `scripts/ceiling_analysis.py` to estimate theoretical max R². Also do domain reasoning (e.g., HOMA = glucose × insulin / 405 → without insulin, ceiling is ~0.6).
5. **Subgroup analysis**: performance by key splits (sex, age bins, feature quantiles)
6. **Feature space visualization**: t-SNE or PCA colored by target, error, key features
7. **Output**: post summary to reporting channel — "data has N samples, M features, target distribution is [X], theoretical ceiling ~[Y], baseline R²=[Z]"

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
- Post to reporting channel: version / R² / what / why / next

### Analysis checkpoints (every 3-5 versions):
- Residual-feature correlations → is there unused signal?
- Subgroup breakdown → which groups improved/worsened?
- Error correlation across models → is there ensemble diversity?
- t-SNE of errors → are failures clustered or dispersed?
- Use findings to generate NEXT hypothesis

### What to try (in order of typical impact):
1. Target transforms (log1p for skewed targets)
2. Feature engineering (domain-specific interactions, ratios)
3. Sample weighting (sqrt(y) for heavy-tailed targets)
4. Hyperparameter tuning (Optuna, 50-100 trials)
5. Diverse model types for blending (trees + linear)
6. Input transforms (QuantileTransformer, StandardScaler)
7. Loss functions (MAE, Huber for robust estimation)

### What usually DOESN'T work (check analysis first):
- More trees when error correlations are >0.95
- Stacking (often leaks — always verify with nested CV)
- Neural nets at n<2000 on tabular data
- Feature selection (trees handle irrelevant features)
- Calibration/stretching predictions post-hoc

## Phase 3: Write Up (last 30%)

When to enter Phase 3:
- 3 consecutive versions with <0.002 improvement AND analysis confirms signal exhausted
- Gap to theoretical ceiling < estimation error
- Time budget running out

What to produce:
1. Comprehensive error analysis + publication-quality figures
2. Paper draft (LaTeX if appropriate) with:
   - The journey (what was tried, what worked)
   - Information-theoretic analysis (why the ceiling exists)
   - Key findings (feature importance, failure modes, subgroups)
3. All figures: consistent style, blue palette, squared panels, clean legends

## Communication Rules

- **Never go >30 min without posting an update** (even "still running fold 3/5")
- **If stuck 30 min with no improvement**: stop and message — "stuck at X, tried Y, hypothesis is Z. Should I pivot?"
- **If 3 experiments show <0.002 gain**: message before trying a 4th
- **If uncertain whether an approach is sound**: ask before spending an hour

## Quality Gates

- **Leakage check**: if R² jumps >0.05 in one version, verify before reporting. Check train vs test R².
- **Overfitting vs ceiling**: if train-test gap >0.20, diagnose (try shallower models, more regularization, or accept info ceiling)
- **Reproducibility**: fixed random seeds, deterministic CV splits, results must be reproducible
- **Figures**: publication-quality from the start. Use `references/style_guide.md` for consistent aesthetics.

## PROJECT_STATE.md Template

Maintain this file in the working directory. Update after every version.

```markdown
# Project State

## Task
[Target variable, metric, dataset location, reporting channel]

## Current Best
[Model A: R²=X (method), Model B: R²=Y (method)]

## Theoretical Ceiling
[Estimated max R² and how it was computed]

## What's Been Tried
| V | Method | R² | Key Finding |
|---|--------|-----|-------------|

## Active Hypotheses
1. [Hypothesis + expected gain + evidence]

## What Analysis Shows
- Residual-feature correlations: [summary]
- Error correlation across models: [summary]
- Subgroup performance: [summary]
- Signal exhaustion: [yes/no + evidence]

## Next Steps
1. [Prioritized list]
```

## Scripts

- `scripts/ceiling_analysis.py` — k-NN neighbor variance ceiling estimator
- `scripts/error_analysis.py` — residual correlations, subgroup breakdown, error correlation matrix
- `scripts/plot_style.py` — shared blue-palette plotting style

See `references/checklist.md` for phase-by-phase checklist.

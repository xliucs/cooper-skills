# ml-hill-climb

An OpenClaw skill for autonomous ML/data science hill-climbing on tabular data. Designed for long-horizon tasks (4-24 hours) with minimal human intervention.

## Why This Skill Exists

This skill was born from two back-to-back ML projects (HOMA-IR prediction, Feb 2026) where the AI agent required heavy human intervention. Here's what went wrong and how this skill prevents it:

### Problems Observed (Without This Skill)

**1. Random model shopping instead of analysis-driven experiments**
The agent spent hours trying models (PyTorch ResNet, FT-Transformer, DCN v2, CatBoost, KNN, Kernel Ridge, SVR...) without analyzing WHY previous models failed. 28 versions were tried before doing proper error analysis at version 25. The error analysis should have been version 3.

**2. Reporting inflated results without sanity checks**
L2 stacking reported R²=0.66 — it was leakage. The agent reported it excitedly without verifying. The human had to be skeptical and push for validation. With proper quality gates, this would have been caught automatically.

**3. Silent grinding for hours**
The agent would run experiments for 2-3 hours without any status update. The human had no idea if progress was being made or if the agent was stuck in a loop. A 30-minute circuit breaker would have surfaced problems early.

**4. No theoretical ceiling analysis**
The agent spent days pushing for R²=0.65 when the theoretical ceiling was R²≈0.61. A ceiling analysis in Phase 1 would have saved hours of futile work and redirected effort toward analysis and writeup.

**5. Catastrophic forgetting across sessions**
After context compaction, the agent would lose track of what had been tried and sometimes re-run failed experiments. The PROJECT_STATE.md pattern solves this.

**6. Analysis only at the end**
Post-hoc analysis (t-SNE, subgroup breakdown, error correlations) produced the most valuable insights of the entire project — but it was done at version 25 out of 28. This skill mandates analysis in every phase.

**7. Poor figure quality on first attempt**
First-pass figures had overlapping legends, inconsistent colors, wrong aspect ratios. The human had to request a redo. The shared plot style and quality standards prevent this.

### How This Skill Fixes These Problems

| Problem | Solution in Skill |
|---------|-------------------|
| Random model shopping | Every experiment requires a stated hypothesis with expected gain and evidence |
| Inflated results | Quality gate: verify any R² jump >0.05 before reporting |
| Silent grinding | 30-min circuit breaker: must post update or message if stuck |
| No ceiling analysis | Phase 1 mandates ceiling_analysis.py before any modeling |
| Catastrophic forgetting | PROJECT_STATE.md maintained every version, read at session start |
| Analysis only at end | Analysis checkpoints every 3-5 versions, drives next hypothesis |
| Poor figures | Shared plot_style.py, publication-quality from version 1 |

## Example Prompt

Here is the recommended prompt for assigning an end-to-end ML task. This prompt is designed to work with the ml-hill-climb skill — the skill will auto-trigger from the task description and load the full protocol.

```
Here's a dataset at /path/to/data.csv with 1078 samples and 25 features.

Task: Predict the column "True_HOMA_IR" (continuous regression). Metric: R².
Reference: a prior paper achieved R²=0.50 using raw wearable time-series with a masked autoencoder.
We only have summary statistics (mean/median/std), not raw time series.

Evaluation: 5-fold × 5-repeat stratified CV, seed 42. Create eval_framework.py and use it for ALL experiments.

Reporting: Post every version update to Telegram group -5164441618.
Format: version number, R², what you tried, why, what's next.

GitHub: push to xliucs/wear-me-dl-v2, update README.md after each version.

Time budget: 8 hours.

Important:
- Start with EDA and ceiling analysis BEFORE trying any models.
- Every experiment must have a hypothesis. If you can't say "I expect +X because Y", don't run it.
- Do error analysis (subgroups, residual correlations, t-SNE) every 3-5 versions, not just at the end.
- If 3 versions show <0.002 improvement and analysis confirms signal exhaustion, stop modeling and write a paper-quality report with publication figures.
- If stuck for 30 minutes with no progress, message me immediately.

Features:
- Demographics (3): age, sex, BMI
- Wearables (15): RHR, HRV, steps, sleep, AZM (each as mean/median/std)
- Blood biomarkers (7): glucose, cholesterol, HDL, LDL, triglycerides, chol/HDL, non-HDL

Two models needed:
- Model A: all features
- Model B: demographics + wearables only (no blood biomarkers)
```

### What This Prompt Does Differently

Compared to the original assignment ("Here's a dataset, predict HOMA-IR, your life depends on it, R²=0.65 baseline"):

1. **Specifies the evaluation protocol upfront** — no more inconsistent CV splits across versions
2. **Sets a time budget** — prevents infinite grinding
3. **Mandates ceiling analysis first** — so I know when to stop
4. **Requires hypotheses** — prevents random model shopping
5. **Includes reporting format** — no more silent hours
6. **Has explicit stuck/exit criteria** — 30-min circuit breaker, 3-version convergence check
7. **References prior work** — gives a realistic baseline instead of an arbitrary target

### What the Agent Will Do (With This Skill)

Phase 1 (~1.5 hours): EDA, baseline (ElasticNet + XGBoost), ceiling analysis (k-NN neighbor variance → R²≈0.61), subgroup analysis, t-SNE visualization. Posts summary: "ceiling is 0.61, baseline is 0.51, biggest signal is glucose."

Phase 2 (~4 hours): Hypothesis-driven experiments. "Log target should help +0.015 because target skewness=2.6." → confirmed. "sqrt(y) weighting should help +0.008 because high-HOMA samples are underweighted." → confirmed. Every 3-5 versions: error analysis checkpoint with residual correlations and subgroup breakdown.

Phase 3 (~2.5 hours): When 3 versions converge, produces comprehensive analysis: information-theoretic ceiling proof, hidden insulin resistant phenotype identification, feature group ablation, publication-quality figures (blue palette, squared panels), paper draft.

## Contents

```
ml-hill-climb/
├── SKILL.md                          # Protocol (auto-loaded by OpenClaw)
├── README.md                         # This file
├── scripts/
│   ├── ceiling_analysis.py           # k-NN neighbor variance R² ceiling estimator
│   ├── error_analysis.py             # Residual correlations, subgroups, error matrix
│   └── plot_style.py                 # Shared blue-palette publication style
└── references/
    └── checklist.md                  # Phase-by-phase checklist
```

## Case Study: HOMA-IR Prediction (Feb 2026)

This skill was developed after the HOMA-IR prediction project, which went through 28 versions over ~24 hours:

- **V1-V6**: Random model shopping. ElasticNet, XGBoost, stacking (leaked), Optuna. Baseline R²≈0.51.
- **V7**: First breakthrough — log target transform (+0.015). This should have been tried in V1 given target skewness=2.6.
- **V11**: Second breakthrough — sqrt(y) sample weighting (+0.008). Found by analysis of tail behavior.
- **V13-V14**: Optuna tuning (+0.005). Diminishing returns starting here.
- **V15-V24**: Ten versions of marginal-to-zero improvement. CatBoost, SMOTER, target decomposition, calibration, stratification. All failed. Should have stopped and analyzed at V17.
- **V25**: Finally did deep error analysis. Found: theoretical ceiling R²=0.614, residual-feature correlations ≈ 0, all signal exhausted. This analysis should have been V3.
- **V26-V28**: Confirmed ceiling. Calibration, stratification, diversity blending — all roads lead to R²≈0.547.
- **Paper**: Wrote NeurIPS-format paper with 5 publication figures in ~2 hours. The analysis was the contribution, not the models.

**Lesson**: The 10 versions of futile grinding (V15-V24) cost ~6 hours. With this skill's protocol, ceiling analysis at V3 and analysis checkpoints every 3-5 versions would have redirected effort to the writeup 6 hours earlier.

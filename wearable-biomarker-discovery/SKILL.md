---
name: wearable-biomarker
description: Discover and validate biomarkers from wearable sensor data (Fitbit, Apple Watch, Oura, Garmin, etc.) for health outcomes (depression, anxiety, stress, sleep disorders). Use when asked to analyze wearable/sensor datasets for health associations, build predictive models from physiological time-series, run biomarker discovery pipelines, or write academic papers on wearable health sensing. Covers the full lifecycle from EDA through statistical testing, machine learning, validation, and paper writing.
---

# Wearable Biomarker Discovery

## Quick Start

1. Read `references/analysis-protocol.md` — the 10-round analysis protocol
2. Run the screening pipeline (Step 1-3 in protocol)
3. Validate findings (Step 4-6)
4. Write up results honestly, including null findings

## Core Principles

**Honest reporting over impressive results.** Wearable-mental-health associations are weak (typical AUC 0.60-0.67). Report null results — they ARE the contribution. Never cherry-pick.

**Effect sizes are small.** Expect ORs of 1.1-1.3 per SD for individual biomarkers. PPV will be 5-10% for low-prevalence outcomes. This is normal and publishable if framed correctly.

**Confirm the ceiling before optimizing.** Before trying fancy models, establish the information-theoretic ceiling with simple methods (logistic regression, k-NN). If LR gets AUC 0.62, XGBoost won't get 0.85.

**Domain knowledge usually wins.** Data-driven discovery (symbolic regression, tsfresh, autoencoders) typically rediscovers what domain experts already know (RHR, sleep variability, step count). This negative result is scientifically valuable.

## Analysis Protocol

Follow the 10-round protocol in `references/analysis-protocol.md`. Summary:

1. **Data cleaning** — Handle NaN, verify N, compute prevalence
2. **Feature screening** — Spearman correlations, FDR correction, effect sizes (OR per SD)
3. **Redundancy pruning** — Correlation matrix, VIF, identify independent signals
4. **Validation split** — Discovery/validation (e.g., 60/40), report replication rate
5. **Advanced methods** — Symbolic regression, CCA, factor analysis, interactions
6. **Clinical utility** — PPV/NPV at realistic thresholds, E-values for confounding
7. **Non-linearity & dose-response** — Polynomial terms, quartile analysis, threshold effects
8. **Cross-system interactions** — Test products/ratios across physiological systems
9. **Representation learning** — PCA vs autoencoder (PCA usually wins)
10. **Honest synthesis** — What's real, what's noise, what's the ceiling

## Known Ceilings & Benchmarks

From DWB (N=4,451) and literature — use these to calibrate expectations:

- **Depression screening (PHQ-9≥10)**: AUC 0.60-0.65, best individual OR ~1.25
- **Anxiety screening (GAD-7≥10)**: AUC 0.63-0.70, typically stronger than depression
- **Top biomarkers** (consistently across studies): RHR mean, sleep variability (CV), step count, RMSSD
- **Symbolic regression**: Fails (R²<0) in weak-signal regimes. Don't spend days on it.
- **Autoencoder vs PCA**: PCA usually matches or beats autoencoders for wearable data
- **Feature replication rate**: Expect 50-75% of discovery features to replicate
- **Demographics alone**: AUC 0.64-0.68 (age, sex, BMI) — wearables must beat this

## Common Pitfalls

- **Sleep SD and CV are redundant** (r>0.90). Pick one (CV is scale-free, prefer it).
- **tsfresh generates mostly redundant features** — 60%+ will correlate >0.95 with simpler features
- **PHQ-9 may be static** (single timepoint) — check before planning longitudinal analyses
- **Multiprocessing breaks** with tsfresh/gplearn in some environments — use n_jobs=1
- **Sample size changes** between raw and clean data need CONSORT-style flow diagrams
- **Questionnaire sum scores** deflate inter-scale correlations via range restriction

## Paper Framing

For wearable biomarker papers with modest effect sizes, frame as:

- **Ceiling establishment**: "We rigorously quantify what passive sensing can and cannot detect"
- **Confirmatory science**: "Data-driven methods validate domain knowledge at scale"
- **Negative results as contribution**: "We tested X sophisticated methods; none beat simple baselines"
- **Clinical honesty**: Report PPV, not just AUC. AUC 0.65 with 5% prevalence = PPV ~7%

## Scripts

- `scripts/screening.py` — Feature screening pipeline (Spearman + FDR + OR per SD)
- `scripts/validate.py` — Discovery/validation split with replication tracking

## References

- `references/analysis-protocol.md` — Full 10-round analysis protocol with code patterns
- `references/statistical-tests.md` — Steiger's test, E-values, bootstrap CIs, CCA, factor analysis

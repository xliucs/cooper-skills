# Wearable Biomarker Discovery

Systematic protocol for discovering digital biomarkers of mental health conditions (depression, anxiety) from consumer wearable sensor data (Fitbit, Apple Watch, Garmin, etc.).

## What This Skill Does

Given a dataset with daily wearable measurements and psychiatric assessments, this skill guides a 10-round analysis pipeline:

1. **Clean data foundation** — Subject-level aggregation, leakage prevention, PHQ/GAD extraction
2. **273+ feature engineering** — Summary stats, temporal dynamics, cross-system interactions, entropy, automated (tsfresh/symbolic regression)
3. **Statistical screening** — Global FDR, confounder adjustment, E-values
4. **Advanced analyses** — Dose-response, Steiger's test (depression vs anxiety), factor analysis, CCA, autoencoders
5. **Validation** — 70/30 discovery-replication split, information ceiling
6. **Integration** — Model ablation, clinical utility assessment
7. **Paper writing** — NeurIPS-quality LaTeX output with honest null reporting

## Key Findings from Development

Developed through 4 rounds of expert review on the DWB dataset (N=4,451, Fitbit, PHQ-9/GAD-7):

- **Two independent biomarkers survive adjustment**: RHR (OR=1.26), Sleep CV (OR=1.16)
- **Three synergistic cross-system interactions**: RHR×cardio_CV, Sleep_CV×cardio_CV, Sleep_CV/steps
- **Anxiety > depression**: 6/10 biomarkers significantly stronger for GAD-7 (Steiger's test, bootstrap confirmed)
- **Non-linear dose-response**: RHR and sleep CV show inflection points
- **Information ceiling**: AUC plateaus at ~0.60-0.65 for single-modality wearables
- **Honest nulls**: Symbolic regression fails (R²<0), autoencoders don't beat PCA, tsfresh 65% redundant

## Usage

```
When asked to discover biomarkers from wearable data, follow SKILL.md phases 1-6.
Each round = new code + new experiments + new findings.
Paper comes LAST, after all analysis rounds.
```

## Origin

Built from the DWB (Depression + Wearable Biomarker) project, 2026. 10 analysis rounds, 4 review cycles, 45+ comments addressed.

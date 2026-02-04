# Phase Checklist

## Phase 1: Understand
- [ ] Load data, check shape, dtypes, missing values
- [ ] Target distribution: mean, std, skewness, percentiles, plot histogram
- [ ] Feature correlations with target (top 10)
- [ ] Create eval_framework.py with FIXED CV splits
- [ ] Run baseline: ElasticNet + XGBoost with default params
- [ ] Run ceiling_analysis.py to estimate theoretical max R²
- [ ] Domain reasoning: is there a formula? What's missing?
- [ ] Subgroup analysis: performance by key splits
- [ ] t-SNE or PCA visualization colored by target
- [ ] Create PROJECT_STATE.md with all findings
- [ ] Post Phase 1 summary to reporting channel

## Phase 2: Experiment (repeat cycle)
- [ ] State hypothesis: "I expect +X because [evidence]"
- [ ] Run experiment
- [ ] Record result in PROJECT_STATE.md
- [ ] git commit + README update
- [ ] Post update to reporting channel
- [ ] Analyze: WHY did it work/not work?
- [ ] Generate next hypothesis from analysis

### Analysis Checkpoints (every 3-5 versions)
- [ ] Residual-feature correlations (signal exhausted?)
- [ ] Subgroup breakdown (who improved? who got worse?)
- [ ] Error correlation across models (diversity?)
- [ ] t-SNE of errors (clustered or dispersed?)
- [ ] Update PROJECT_STATE.md with analysis findings

## Phase 3: Write Up
- [ ] Comprehensive post-hoc analysis
- [ ] Publication-quality figures (consistent style)
- [ ] Paper draft or technical report
- [ ] Final git commit + push
- [ ] Post final summary to reporting channel

# Phase Checklist

## Phase 1: Understand
- [ ] Load data, check shape, dtypes, missing values
- [ ] Target distribution (regression: skewness, outliers; classification: class balance)
- [ ] Feature correlations with target (top 10)
- [ ] Create eval_framework.py with FIXED CV splits
- [ ] Run baselines (linear model + tree model)
- [ ] Ceiling analysis: k-NN neighbor variance (regression) or domain reasoning
- [ ] Prior work comparison: what scores has previous work achieved? Why?
- [ ] Subgroup analysis: performance by key categorical/continuous splits
- [ ] Feature space visualization: t-SNE or PCA colored by target and error
- [ ] Create PROJECT_STATE.md with all findings
- [ ] Post Phase 1 summary to reporting channel

## Phase 2: Experiment (repeat cycle)
- [ ] State hypothesis: "I expect +X because [evidence]"
- [ ] Run experiment
- [ ] Record result in PROJECT_STATE.md
- [ ] git commit + README update
- [ ] Post update to reporting channel (version / score / what / why / next)
- [ ] Analyze: WHY did it work/not work?
- [ ] Generate next hypothesis from analysis

### Analysis Checkpoints (every 3-5 versions)
- [ ] Residual-feature correlations (regression) or confusion patterns (classification)
- [ ] Subgroup breakdown (who improved? who got worse?)
- [ ] Error correlation across models (is there diversity for ensembling?)
- [ ] t-SNE of errors (clustered failures = fixable, dispersed = information limit)
- [ ] Update PROJECT_STATE.md with analysis findings
- [ ] Check: are we approaching the ceiling? Should we enter Phase 3?

## Phase 3: Write Up
- [ ] Final comprehensive error analysis
- [ ] Publication-quality figures (use plot_style.py)
- [ ] Technical report or paper draft
- [ ] Feature importance / ablation study
- [ ] Subgroup disparity analysis
- [ ] Ceiling analysis discussion (why can't we do better?)
- [ ] Final git commit + push
- [ ] Post final summary to reporting channel

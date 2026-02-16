# Domain Knowledge: Wearable Biomarkers for Depression

## Established Associations (meta-analyses exist)

| Association | Key References | Effect Size |
|---|---|---|
| Reduced HRV (RMSSD) → depression | Koch 2019, Kemp 2010 | Small-medium |
| Reduced sleep duration → depression | Baglioni 2016 | Small |
| Altered REM% → depression | Baglioni 2016 | Small-medium |
| Reduced physical activity → depression | Schuch 2018, Pearce 2022 | Small-medium |
| Poor sleep efficiency → depression | Scott 2021 | Small |
| Elevated RHR → depression | Kemp 2010 | Small |
| Later sleep midpoint → depression | Walch 2016 | Small |

## Known but Less Documented

- Sleep fragmentation (WASO) → depression
- Deep sleep reduction → depression
- Step count reduction → depression (Choi 2019, Mendelian randomization)
- Later bedtime → depression

## Feature Aliasing (NOT independent signals)

- sleep_time_minutes ≈ duration_minutes ≈ sleep_duration_hours_calculated
- sleep_start_time_hour ≈ sleep_start_time_hour_decimal
- sleep_midpoint_hour ≈ f(sleep_start, sleep_end)
- Sleep SD and CV are r=0.93 correlated — report only CV (scale-invariant)

## Benchmark Effect Sizes (Jacobson et al. 2019, npj Digital Medicine)

- Wearable → depression: r = 0.10–0.30
- R² = 0.05–0.15 for single-modality
- Anything above r = 0.40 is suspicious

## Sample Size vs AUC (empirical relationship)

| Study | N | AUC | Modality |
|---|---|---|---|
| Jacobson 2019 | 31 | 0.89 | Fitbit + phone |
| Chikersal 2021 | 138 | 0.78 | Fitbit + phone |
| Moshe 2021 | 149 | 0.60 | Phone only |
| DWB analysis | 4,451 | 0.60-0.65 | Fitbit only |

The inverse N-AUC relationship is consistent with small-sample overfitting.

## Why Anxiety > Depression for Wearables

GAD involves sustained sympathetic activation (elevated HR, reduced HRV, muscle tension) directly measurable by wearables. Depression is heterogeneous: DSM-5 includes both insomnia AND hypersomnia, both psychomotor agitation AND retardation, both appetite increase AND decrease. This symptom polarity creates within-group heterogeneity that attenuates aggregate biomarker signals.

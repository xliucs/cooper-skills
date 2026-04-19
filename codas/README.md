# codas — AI Co-Data-Scientist for Wearable Biomarker Discovery

A **Claude Code skill** that reproduces the [CoDaS pipeline](https://arxiv.org/abs/2604.14615) (Kim et al., 2026) — a multi-agent system for autonomous discovery and validation of digital biomarkers from wearable sensor + clinical tabular data.

Drop the skill in, point it at a dataset and a clinical target, and Claude Code runs the full six-phase loop end-to-end: profile → hypothesize → statistically test → adversarially validate → mechanistically annotate → ship a polished HTML report.

## What this is for

If you have:
- A tabular wearables dataset (one row per participant, or per participant-wave) with features like resting heart rate, HRV, steps, sleep architecture, screen-time, GPS mobility, etc., **and**
- A clinical target column (PHQ-8, HOMA-IR, HbA1c, anything continuous or binary),

…then a single prompt like:

```
/codas dataset=wear_me.parquet target=HOMA_IR id=participant_id \
       exclude=insulin,glucose demographics=age,sex,bmi
```

…will hand back a folder containing:
- **`report.html`** — primary deliverable. Single self-contained file, Apple/Google design quality. Hero stats + animated Δ R² gauge, forest plot of candidate effect sizes with 95% CIs, validation-battery heatmap (candidates × 11 tests), per-candidate cards with verdict badges and expandable mechanism/debate detail, discovery-trajectory chart (Jaccard convergence + R² progression by round), cohort profile, methods accordion, honest limitations, reproducibility footer with Fact Sheet hash. Light/dark mode, mobile-responsive, WCAG AA, prefers-reduced-motion respected. Open in any browser; no server needed.
- `summary.json` — machine-readable digest of validated candidates and CV metrics.
- `phase_C/verdicts.json` — per-candidate pass/fail across all 11 validation tests.
- `phase_E/fact_sheet.json` — the deterministic ground-truth numbers all prose was copied from.
- `phase_E/reviewer_flags.json` — final QC findings from the Reviewer agent.
- `audit.log` — every agent invocation, every Python subprocess.
- `report.md` — *only if `--with-paper` is set.* Manuscript-format markdown for arXiv-style submission.

## About the paper

**CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors**
Kim, Y., Rahman, S., Schmidgall, S., Park, C., Heydari, A.A., Metwally, A.A., Yu, H., Liu, X., Xu, X., Yang, Y., Xu, M.A., Zhang, Z., Breazeal, C., Althoff, T., Sirkovic, P., Rendulic, I., Pawlosky, A., Stroppa, N., Gottweis, J., Vedadi, E., Karthikesalingam, A., Kohli, P., Natarajan, V., Malhotra, M., Patel, S., Park, H.W., Palangi, H., McDuff, D. (2026)
arXiv:[2604.14615](https://arxiv.org/abs/2604.14615) · 28 authors across Google Research, Google DeepMind, Google Cloud AI, MIT.

### The headline claim

CoDaS treats biomarker discovery as a structured multi-agent workflow rather than a monolithic LLM call. Across **9,279 participant-observations** in three cohorts (DWB, GLOBEM, WEAR-ME) it surfaced **41 mental-health and 25 metabolic candidate biomarkers**, each subjected to an internal validation battery operationalizing four dimensions across **11 deterministic checks**.

Highlight findings the paper attributes to the pipeline's autonomous discovery:

| Cohort | Candidate | ρ (Spearman) | p |
|--------|-----------|------|------|
| DWB (PHQ-8, N=7,497) | Main sleep duration variability | 0.252 | < 0.001 |
| DWB | Nocturnal social-app usage | 0.246 | < 0.001 |
| DWB | Night/day social-media ratio (composite) | 0.222 | < 0.001 |
| GLOBEM (PHQ-4, N=704) | Sleep onset time variability | 0.126 | < 0.001 |
| WEAR-ME (HOMA-IR, N=1,078) | Cardiovascular fitness index = steps / RHR (composite) | −0.374 | < 0.001 |
| WEAR-ME | AST/ALT ratio (composite) | −0.375 | < 0.001 |

Adding the top CoDaS-selected features beyond demographics yielded **ΔR² = 0.040 on PHQ-8** and **ΔR² = 0.021 on HOMA-IR (wearables only)** under participant-level 5-fold CV.

### Why a multi-agent pipeline (and not just one big LLM call)

Three risks dominate naive LLM-driven biomarker hunts:

1. **Leakage.** A combinatorial feature space contains many algebraic transforms of the target. CoDaS handles this with explicit `excluded_features` lists, a |ρ| > 0.85 construct-overlap gate, label isolation (LLM agents only see summary statistics, never raw labels), and participant-level `GroupKFold`.
2. **Hallucinated numbers.** Prose-writing LLMs invent sample sizes, R²s, and validation counts. CoDaS computes a **Fact Sheet** — a flat dictionary of every reportable number — *before* any section is drafted, and a deterministic numeric-verification pass scans the draft against it.
3. **Spurious correlations.** Population-scale data makes weak-signal noise statistically significant. CoDaS counters this with an 11-test validation battery and a parallel **Critic ↔ Defender debate** that must adjudicate every survivor.

### The six phases (what this skill orchestrates)

```
A. Profiling + Literature Grounding   Scout agent + Researcher ensemble (Lit search, Mechanism extraction)
B. Iterative Discovery Loop            Hypothesis → Stat runner → ML runner → Critic (looped, GapChecker convergence)
C. Adversarial Validation              11-test battery + parallel Critic vs Defender debate
D. Mechanism / Novelty / Strategy      Three parallel agents annotate each survivor
E. Report Writing & Assembly           Fact Sheet → section writers → numeric verifier
F. Human Feedback (optional)           Plausibility-gap / conflict / stagnation triggers
```

### The 11-test validation battery

Organized into four dimensions:

**Replication** (1) Independent held-out replication.
**Stability** (2) Permutation test, (3) Bootstrap CI, (4) Leave-one-out influence.
**Robustness** (5) Subgroup consistency (sex × age), (6) Method triangulation (Spearman/Pearson/Kendall), (7) **Construct-validity hard gate** (auto-reject if |ρ|>0.85), (8) Causal robustness (partial correlation residualizing demographics + prior validated biomarkers).
**Discriminative power** (9) **Construct-independence hard gate**, (10) **CI-consistency hard gate**, (11) AUC ≥ 0.55 floor.

**Verdict logic:** ≥70% pass + all core tests {1, 2, 3, 10} → `validated`. 40–70% → `conditional`. Else `rejected`. Hard gates 7, 9, 10 auto-reject regardless of pass rate.

## Why this skill is for Claude Code specifically

CoDaS is a multi-agent system. Claude Code is well-suited because it natively offers:

- **Subagent dispatch** via the `Agent` tool — used for Scout / Hypothesis / Critic / Defender / Mechanism / Novelty / Strategy / section-writer roles.
- **Parallel agent invocation** in a single message — required for the Critic↔Defender debate so neither sees the other's argument first.
- **Bash + filesystem tools** — required for the deterministic Python statistical runners that enforce the "LLMs never emit final numbers" invariant.
- **Slash-command activation** (`/codas`) and natural-language triggers ("discover biomarkers in this dataset").

The skill makes the orchestration declarative: trigger words, required inputs, phase ordering, validation thresholds, and quality gates are all spelled out in `SKILL.md` so Claude Code can run the pipeline without further prompting.

## Install

Clone this repo and copy the skill into your Claude Code skills directory:

```bash
git clone https://github.com/xliucs/cooper-skills.git
mkdir -p ~/.claude/skills
cp -r cooper-skills/codas ~/.claude/skills/
```

Then in any Claude Code session:

```
/codas
```

…or just say "discover biomarkers in `<dataset>` predicting `<target>`" and Claude Code will pick up the skill from its description.

## Use

Minimal invocation — Claude Code will ask for any missing input:

```
/codas
```

Or pass everything up front:

```
Run CoDaS on /data/wear_me.parquet predicting HOMA_IR.
Participant ID column: participant_id.
Exclude from features: insulin, glucose, hba1c.
Demographics for baseline: age, sex, bmi.
Output to ./codas_homair_run/.
```

For a fast smoke test (smaller permutation + bootstrap counts, fewer rounds):

```
/codas --quick dataset=... target=...
```

## Knobs (defaults from the paper)

| Knob | Default | Where it's used |
|------|---------|------------------|
| Discovery rounds | 4 | Phase B; Jaccard convergence at 0.80 typically exits at round 3 |
| Permutation iterations | 1000 | Test 2 |
| Bootstrap iterations | 1000 | Test 3 |
| CV folds | 5 | `GroupKFold` keyed by participant ID |
| FDR α | 0.05 | Benjamini–Hochberg, per-round across full univariate family |
| Hard-gate threshold | \|ρ\| > 0.85 | Test 7 + leakage construct-overlap screen |
| Min validated effect size | \|ρ\| ≥ 0.10 | Verdict downgrade rule |
| Discriminative AUC floor | 0.55 | Test 11 |
| Composite share target | ≥40% of proposals | Hypothesis-agent prior (paper: composites |ρ|=0.28 vs raw 0.21) |
| Imputation strategies | median, KNN-5, iterative | Phase E sensitivity check (Δρ < 0.01 stability) |

**Flags:**
- `--quick` — perm/bootstrap → 200, rounds → 2. Draft only.
- `--with-paper` — also emit `report.md` manuscript draft.
- `--no-literature` — skip Researcher ensemble (ablation).
- `--seed=<int>` — fix RNG seed (default 42).

## Expected runtime & cost

Per the paper (DWB N=7,497, on a single machine):
- **Wall clock**: 6–8.5 hours end-to-end.
- **Tokens**: ~7M in / ~200K out.
- **LLM cost**: ~$4.
- **Smaller cohorts** (WEAR-ME-class N~1k): 2–3 hours.

**Models**: high-reasoning model for Scout / Hypothesis / Critic / Defender / Mechanism / Section-writer roles; low-latency model for repeated lower-stakes tasks (paper verifier, numeric verifier, per-candidate annotation). Within Claude Code, this is the default Opus + Sonnet/Haiku Agent dispatch — no extra config needed.

## Outputs

```
<output_dir>/
├── report.html                  # PRIMARY DELIVERABLE — single-file interactive report
├── report.md                    # OPTIONAL — only with --with-paper flag
├── summary.json                 # digest: n_validated, top candidates, CV metrics, reviewer flags
├── audit.log                    # every agent + every Python subprocess
├── phase_A/
│   ├── schema.json              # dataset profile from Scout
│   ├── eda.json                 # missingness, correlations, target distribution
│   └── literature_priors.json   # established biomarkers + pathways from Researcher
├── phase_B/
│   ├── round_<k>/stats.parquet  # per-round candidate stats
│   └── convergence.json         # Jaccard-per-round + R² progression
├── phase_C/
│   └── verdicts.json            # per-candidate × per-test results + Critic/Defender adjudications
├── phase_D/
│   └── annotations.json         # mechanism / novelty / strategy notes
└── phase_E/
    ├── fact_sheet.json              # ground-truth numbers (source of truth for report.html)
    ├── imputation_sensitivity.json  # Δρ across {median, KNN-5, iterative}
    ├── reviewer_flags.json          # final QC pass — hallucinations / stat errors / contradictions
    └── corrections.log              # what the numeric verifier auto-corrected
```

### What the HTML report contains

1. **Hero** — three big stat cards: validated count, Δ R², top candidate.
2. **Executive summary** — 3 plain-language sentences.
3. **Top candidates** — sortable cards with verdict badges (✅ validated / ⚠️ conditional / ❌ rejected), novelty stars, ρ + 95% CI, expandable mechanism + 11-test detail + Critic↔Defender debate snippets.
4. **Validation battery heatmap** — interactive: candidates × 11 tests, click-to-filter.
5. **Performance** — bar chart of demographics-only vs +biomarkers, with per-fold scatter.
6. **Discovery trajectory** — Jaccard convergence + R² progression by round.
7. **Cohort profile** — demographic donut, target histogram, missingness bar.
8. **Methods** — accordion with the 6-phase pipeline diagram (inline SVG), validation battery spec, leakage guards.
9. **Limitations** — softer-toned section; auto-includes the standard CoDaS caveats.
10. **Reproducibility** — Fact Sheet hash, run config, seed, model versions.
11. **Footer** — collapsible BibTeX, "Generated by CoDaS skill — Claude Code".

Built to satisfy the seven user-study quality dimensions from paper §6.1 (Novelty, Soundness, Presentation, Plausibility, Statistical Validity, Reproducibility, Limitations) — the dimensions on which CoDaS scored 3.1–4.1 vs 1.3–2.6 for baselines in blind expert review.

## Limitations and honest caveats

The paper itself is careful to flag, and this skill preserves, the following:

- **Effect sizes are modest** (typical |ρ| = 0.15–0.30). Population-scale wearable phenotyping is an inherently weak-signal regime.
- **All findings are hypothesis-generating.** Surviving the internal battery is *not* prospective external validation, regulatory endorsement, or clinical proof.
- **GLOBEM endpoint selection was data-driven** (PHQ-4 chosen for higher coverage than BDI-II), introducing optimism bias for that cohort.
- **The 11 checks are not orthogonal** — they're a structured audit, not 11 independent significance tests.
- **Construct convergence ≠ replication.** The paper observes circadian-instability features in two depression cohorts but with different operationalizations; this is suggestive, not confirmatory.

The skill inherits these caveats and surfaces them in the auto-generated `Limitations` section of `report.md`.

## Citation

If this skill helps your research, please cite the paper:

```bibtex
@article{kim2026codas,
  title={CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors},
  author={Kim, Yubin and Rahman, Salman and Schmidgall, Samuel and Park, Chunjong and Heydari, A. Ali and Metwally, Ahmed A. and Yu, Hong and Liu, Xin and Xu, Xuhai and Yang, Yuzhe and Xu, Maxwell A. and Zhang, Zhihan and Breazeal, Cynthia and Althoff, Tim and Sirkovic, Petar and Rendulic, Ivor and Pawlosky, Annalisa and Stroppa, Nicolas and Gottweis, Juraj and Vedadi, Elahe and Karthikesalingam, Alan and Kohli, Pushmeet and Natarajan, Vivek and Malhotra, Mark and Patel, Shwetak and Park, Hae Won and Palangi, Hamid and McDuff, Daniel},
  journal={arXiv preprint arXiv:2604.14615},
  year={2026}
}
```

## License

Skill packaging: same as parent repo. The CoDaS methodology described in the upstream paper is © 2026 Google, CC-BY-4.0.

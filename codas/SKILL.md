---
name: codas-biomarker-discovery
description: Reproduce the CoDaS (AI Co-Data-Scientist) multi-agent pipeline from Kim et al. 2026 (arXiv:2604.14615) for autonomous biomarker discovery from wearable sensor / clinical tabular data. Activates on "/codas", "discover biomarkers", "find digital biomarkers", "run CoDaS on this dataset", or any request to autonomously identify, validate, and report candidate biomarkers from a wearables + clinical-target dataset. Implements the six-phase loop (profiling → hypothesis → stat/ML → adversarial validation → mechanism/novelty → report), the 11-test validation battery across four dimensions (replication, stability, robustness, discriminative power), and the leakage-prevention guards. Default deliverable is a production-grade interactive HTML report grounded in a deterministic Fact Sheet; manuscript-style markdown is optional.
---

# CoDaS — AI Co-Data-Scientist for Wearable Biomarker Discovery

Faithful reproduction of the pipeline in Kim, Y. et al. "CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors" (arXiv:2604.14615, 2026). One prompt in → six-phase autonomous discovery → validated candidate biomarkers + a polished interactive HTML report (Apple/Google production quality) grounded in a deterministic Fact Sheet.

**Recommended models** (matching the paper): a high-reasoning model (Claude Opus, GPT-5-class, or Gemini 3 Pro) for Scout / Hypothesis / Critic / Defender / Mechanism / Report-section roles; a low-latency model (Claude Haiku, Gemini 3 Flash) for repeated lower-stakes tasks (paper verifier, numeric verifier, per-candidate annotation). Within Claude Code, this is the default Opus-and-Sonnet/Haiku Agent dispatch.

**Expected runtime / budget** (per the paper, on DWB N=7,497): ~6–8.5 hours wall-clock, ~7M input tokens, ~200K output tokens, ~$4 LLM cost on a single machine. WEAR-ME-class smaller cohorts (N~1k) finish in 2–3 hours.

## When to invoke

Activate when the user asks any of:
- "discover biomarkers in `<dataset>`"
- "run CoDaS on `<dataset>` predicting `<target>`"
- "/codas `<args>`"
- "find digital biomarkers for `<outcome>` using these wearable features"
- Any task that pairs a wearable / physiological tabular dataset with a clinical target and asks for autonomous candidate-feature discovery + validation.

## Required inputs (ask for any missing — do not invent)

1. **Dataset path** (CSV / Parquet / Feather). One row per participant or participant-wave observation.
2. **Target column** (e.g., `PHQ8`, `HOMA_IR`). Continuous → regression; binary/ordinal → classification.
3. **Participant ID column** (for stratified splits; required even if one row per participant — used for repeated-measures handling).
4. **Excluded-from-features list**: the target plus any direct clinical proxies (e.g., for HOMA-IR exclude `glucose`, `insulin`; for PHQ-8 exclude `BDI-II`, `PHQ-2`). If the user omits this, *propose* a list from the schema and ask for confirmation before proceeding.
5. **Demographic covariates**: columns for the demographics-only baseline model (default: age, sex, BMI if present).
6. **Output directory** for the run (default: `./codas_run_<timestamp>/`).

If any of (1)–(4) is missing, ask before starting. If the user says "just figure it out," use the Scout agent (Phase A) to propose them and request a single confirmation.

## Pipeline overview

Six phases, executed in order. Track each as a TaskCreate item; mark in-progress on entry, completed on exit. Each phase persists artifacts to `<output_dir>/phase_<X>/` so the run is auditable.

```
A. Data Profiling + Literature Grounding   [Scout, Researcher ensemble]
B. Iterative Discovery Loop                 [Hypothesis, Stat, ML, Critic — looped until GapChecker convergence]
C. Adversarial Validation                   [Critic vs Defender debate; 11-test battery]
D. Mechanism + Novelty Assessment           [Mechanism, Novelty, Strategy agents]
E. Report Writing & Assembly                [Fact Sheet → Writer subagents → Numeric Verifier]
F. Human Feedback (optional)                [present draft, accept guidance, optionally re-loop]
```

Run each LLM-driven role via the **Agent tool** (`subagent_type: general-purpose` unless noted). All numerical computation goes through deterministic Python via **Bash** — never let an LLM emit final numbers. This is the paper's core integrity invariant.

---

## Phase A — Data Profiling + Literature Grounding

### A1. Scout agent
Spawn an Agent with a self-contained prompt that:
- Reads the dataset (use `pandas` via Bash; do not load via Read for large files).
- Emits to `<output_dir>/phase_A/schema.json`: column names, dtypes, missingness %, n_unique, simple summary stats, detected ID column, candidate target columns.
- Validates the user-supplied target exists and has ≥80% coverage; if not, surface a warning.
- Proposes an `excluded_features` list (target + Spearman |ρ|>0.85 with target candidates).
- Returns: schema summary + proposed exclusions for confirmation.

### A2. Deterministic EDA runner
Run a Python script that writes `<output_dir>/phase_A/eda.json`:
- Per-column missingness, mean, std, IQR, n_unique.
- Pairwise Spearman correlation matrix among numeric features (capped at 200 cols by variance).
- Target distribution (histogram bins, skew, kurtosis).
- Subgroup counts (sex, age decade) for later subgroup-consistency tests.

### A3. Researcher ensemble
Spawn an Agent (or sequence) with `WebSearch` and `WebFetch` access — both are built-in Claude Code tools, no setup required.

**Search strategy (in priority order):**
1. **arXiv** — best signal-to-noise. `WebFetch` directly on `https://arxiv.org/abs/<id>` or `https://arxiv.org/list/q-bio/recent`. Returns clean abstract + metadata.
2. **PubMed** — `WebSearch` with `"<topic> site:pubmed.ncbi.nlm.nih.gov"` then `WebFetch` each hit. No fielded MeSH search natively, so use plain-English queries.
3. **Google Scholar** — `WebSearch` works for ranking; some result pages are Cloudflare-protected. If `WebFetch` returns a 403 or a challenge page, fall through to Playwright (per global CLAUDE.md: `NODE_PATH=/opt/homebrew/lib/node_modules node script.js` with headless Chrome) — never give up on a 403.

**Optional augmentation (recommended for production runs):** if a PubMed MCP server is installed (e.g., `mcp__pubmed__search` for proper E-utilities access with MeSH terms, date ranges, and author filters), prefer it over the plain `WebSearch` path. Check `claude plugin marketplace list` for available servers before falling back to web search.

**What to collect:**
- **Literature searcher**: 3–5 queries combining `<target>` + {"wearable", "digital biomarker", "passive sensing", "actigraphy", "PPG"}. Collect top 20 abstracts.
- **Paper verifier**: before citing anything, confirm the paper exists — `WebFetch` the DOI or arXiv URL and check it resolves to a real paper, not a 404 or a hallucinated title.
- **Mechanism extractor**: produce a structured JSON of `{established_biomarkers: [...], pathways: [...], known_confounders: [...]}` saved to `<output_dir>/phase_A/literature_priors.json`.

Pass `literature_priors.json` to all downstream agents as biological-anchor context.

**Hard limits to respect:**
- No authenticated databases (Embase, Scopus, Web of Science) — those need user-supplied credentials and are out of scope for this skill.
- Cap total search calls at ~30 per run to keep wall time bounded; if more depth is needed, raise it explicitly.

---

## Phase B — Iterative Discovery Loop

Run for at most **N=4 rounds** by default (paper used 4; converged by round 3 on DWB). Each round:

### B1. Hypothesis agent
Agent invocation, given `schema.json` + `literature_priors.json` + prior-round results:
- Propose 10–30 candidate features. Each entry: `{name, formula (Python expression over existing columns), rationale, mechanistic_link (cite literature_priors), est_effect_direction, type ∈ {raw, ratio, variability, composite}}`.
- **Bias toward composites.** The paper found composites have mean |ρ|=0.28 vs 0.21 for their constituent raw features — a real signal-amplification effect. Aim for ≥40% of proposals to be composites or variability summaries, not raw columns.
- Mix three types:
  (a) **Raw** existing columns (anchor set).
  (b) **Ratios / composites** motivated by literature (canonical examples from the paper: `steps / resting_hr` = cardiovascular fitness index; `AST / ALT` = De Ritis ratio; `night_app_usage / day_app_usage` = night/day ratio; `hedonic_apps / productivity_apps` = anhedonia proxy; `HRV / RHR` = autonomic balance; `albumin / globulin`).
  (c) **Variability summaries** (rolling SD, IQR, MAD, polyphasic-sleep-percentage, sleep onset SD) — circadian-instability features dominate the depression results.
- Forbidden: any feature whose formula touches `excluded_features` directly or via algebraic transform of the target.

### B2. Statistical runner (deterministic Python)
For each proposed candidate:
- Compute the feature column from the formula (use `df.eval` with allowlisted functions).
- Spearman ρ + p-value vs target; Pearson ρ; bootstrap 95% CI (1000 resamples, fixed seed).
- BH-FDR correct p-values across the round's full family (α=0.05).
- Persist to `<output_dir>/phase_B/round_<k>/stats.parquet`.

### B3. ML runner (deterministic Python)
- 5-fold cross-validation, **stratified at participant-ID level** (use `GroupKFold` keyed by participant ID).
- Models: Ridge regression (regression target) or Logistic Regression + Gradient Boosting (classification).
- Metrics: CV R² (regression) or CV AUC (classification), with per-fold SD.
- Compare: demographics-only baseline vs demographics + top-K candidates (K=5,10,15). Report ΔR² or ΔAUC.

### B4. Critic agent — pre-validation pruning
Agent given (B2 + B3 outputs):
- Apply **construct overlap analysis**: drop any candidate whose Spearman |ρ| > 0.85 with any `excluded_features` column.
- Drop candidates failing FDR-corrected p > 0.05.
- Drop candidates whose composite formula uses only excluded variables.
- Output a shortlist for Phase C.

### B5. GapChecker — Jaccard-based convergence (deterministic)
Decide whether to continue looping. Per the paper, convergence is **Jaccard-based** on the candidate shortlist:
- Compute `jaccard_k = |shortlist_k ∩ shortlist_{k-1}| / |shortlist_k ∪ shortlist_{k-1}|`.
- **Stop** (declare convergence) iff `jaccard_k ≥ 0.80` (≥80% overlap with prior round) OR k ≥ N.
- Otherwise also stop early if best CV R² (or AUC) improved by < 0.003 over round k-1 AND no candidate added that brings new mechanistic family.
- Continue otherwise.

The paper converged at round 3 of 4 on DWB. Log `jaccard_k` to `<output_dir>/phase_B/convergence.json` for the report.

### B6. Holistic prioritization (5 axes — paper §2.4)
Beyond pure stat significance, score each candidate on five axes (1–5 each) and rank by composite. These axes also drive what gets featured in the HTML report:
1. **Statistical Validity** — effect size, FDR-adjusted p, CV performance contribution.
2. **Clinical Plausibility** — mechanism agent score (Phase D).
3. **Originality** — novelty agent score (Phase D); composites and Emerging★★ candidates rank higher than re-discoveries of Established markers.
4. **Generalizability** — pass rate on subgroup-consistency tests across sex × age stratifications.
5. **Interpretability** — penalize black-box composites with no plausible biological label; reward physiologically named ratios (cardiovascular fitness index, autonomic balance ratio).

---

## Phase C — Adversarial Validation (the 11-test battery)

For each converged candidate, run all 11 checks in `<output_dir>/phase_C/battery.py` (deterministic). The four dimensions and exact specs:

### Dimension 1 — Replication
1. **Independent replication.** Held-out confirmation set (participant-level split, 20% of participants, N≥20). Spearman on held-out. Pass if p < 0.05 AND sign matches discovery sign. For repeated-measures cohorts, retain one randomly chosen row per participant in the confirmation set.

### Dimension 2 — Stability
2. **Permutation test.** 1000 label-permuted resamples → empirical null. Pass if observed |ρ| > 95th percentile of null.
3. **Bootstrap stability.** 1000 bootstrap resamples; 95% CI must not straddle zero.
4. **Leave-one-out influence.** For N up to 2000, drop one participant at a time; for larger N, drop random 1% chunks 100 times. Pass if sign of ρ never flips.

### Dimension 3 — Robustness
5. **Subgroup consistency.** Split cohort in halves: by sex (female vs male) AND by median split of age. Within each half, recompute ρ; pass if sign is consistent across all subgroups (Simpson's-paradox guard).
6. **Method triangulation.** Recompute association using Pearson and Kendall's τ. Pass if all three methods (incl. Spearman) yield p < 0.05.
7. **Construct validity hard gate** — *automatic rejection if violated*. Reject if |ρ| > 0.85 with target for N > 30 (adaptive: > 0.90 for 20<N≤30; > 0.95 for N≤20). Indicates an undetected tautology.
8. **Causal robustness.** Partial Spearman residualizing the candidate against demographic covariates AND any previously-validated biomarker. Pass if residualized ρ remains significant (p < 0.05).

### Dimension 4 — Discriminative power
9. **Construct independence hard gate** — *automatic rejection if violated*. For composite features, check whether each constituent column independently correlates with target at |ρ| > 0.5; classify as `independent` / `proxy` / `compositional`. Reject `proxy` and `compositional`.
10. **CI consistency hard gate** — *automatic rejection if violated*. Sign of point estimate must agree with sign of bootstrap CI midpoint.
11. **Discriminative power.** Build a univariate model (logistic regression on binarized target at median for regression endpoints). Pass if AUC ≥ 0.55.

### Verdict logic (apply in this order)
- **Auto-reject** if any of tests 7, 9, 10 fail (hard gates).
- **Auto-reject** if tests 1, 2, 3 *all* fail simultaneously.
- **`validated`** if pass rate ≥ 70% of applicable tests AND all of {1, 2, 3, 10} pass.
- **`conditional`** if pass rate 40–70%, OR `validated` was downgraded due to marginal effect size (|ρ| < 0.10).
- **`rejected`** otherwise.

### C2. Critic ↔ Defender debate (LLM)
For every candidate that survives the deterministic battery, spawn **two parallel agents in a single message**:
- **Critic agent**: argue against retention. Look for confounders, literature contradictions, suspicious construction. Produce 3–5 specific objections.
- **Defender agent**: argue for retention. Cite mechanistic plausibility from `literature_priors.json` and the battery results.

Then synthesize a one-paragraph adjudication (Orchestrator role — you do this directly, not a subagent). Downgrade `validated` → `conditional` if any Critic objection is unrebutted by empirical evidence in the Fact Sheet.

Persist all per-candidate verdicts + per-test results to `<output_dir>/phase_C/verdicts.json`.

---

## Phase D — Mechanism, Novelty, Strategy

For each candidate with verdict `validated` or `conditional`, run three parallel agents:
- **Mechanism agent**: 1–2 sentence physiological rationale grounded in `literature_priors.json` (cite specific entries).
- **Novelty agent**: classify as `Established` (well-known correlation), `Supported★` (mechanism known, this operationalization new), or `Emerging★★` (limited prior evidence). Justify with 1 sentence + citation.
- **Strategy agent**: 1 sentence on translational value (e.g., "noninvasive surrogate for clinical assay X enabling longitudinal monitoring").

Persist to `<output_dir>/phase_D/annotations.json`.

---

## Phase E — Report Writing & Assembly (HTML-first)

The default deliverable is a **production-grade interactive HTML report** — Apple/Google design quality, single self-contained file. Markdown manuscript is optional (`--with-paper` flag).

### E1. Sensitivity analyses (deterministic, run before Fact Sheet)
Per paper §5.1:
- **Imputation sensitivity**: re-run top-K candidate stats under 3 imputation strategies (median, KNN with k=5, IterativeImputer). Pass if max Δρ across strategies < 0.01. Persist to `<output_dir>/phase_E/imputation_sensitivity.json`.
- **Threshold sensitivity**: re-run candidate selection at default (`p<0.05, |ρ|≥0.20`) and lenient (`p<0.10, |ρ|≥0.10`) thresholds. Report which candidates survive both vs. only one.

### E2. Build the Fact Sheet (deterministic, MUST run before any prose)
Generate `<output_dir>/phase_E/fact_sheet.json` — a flat key→value dictionary computed from pipeline state. Required keys:
- **Cohort**: `n_participants`, `n_observations`, `n_features_initial`, `n_features_after_screening`, `cohort_name`, `monitoring_duration_days`, `missingness_pct`, demographic breakdown.
- **Target**: `target_name`, `target_mean`, `target_sd`, `target_n_nonmissing`, `target_distribution_bins` (10 bins for histogram).
- **Discovery**: `n_candidates_proposed`, `n_candidates_passing_screen`, `n_validated`, `n_conditional`, `n_rejected`, `n_rounds_run`, `convergence_round`, `jaccard_per_round`.
- **Performance**: `cv_r2_demographics_only` (mean+SD), `cv_r2_demographics_plus_top5` (mean+SD), `delta_r2`, per-fold CV scores for the demographics-only and full models, `cv_auc_*` analogs for classification targets.
- **Per-candidate**: for each non-rejected candidate `{candidate_id: {name, formula, type, rho_spearman, rho_pearson, tau_kendall, ci_lo, ci_hi, adj_p, perm_pvalue, n_tests_passed, n_tests_applicable, verdict, mechanism, novelty_tier, strategy_note, holistic_score, sensitivity_stable, debate_outcome}}`.
- **Quality gates**: `quality_gate_flags` (see E4).
- **Pipeline meta**: `run_duration_sec`, `model_versions`, `run_timestamp`, `seed`.

### E3. Section writers (LLM agents, pass Fact Sheet as context)
Spawn agents in parallel to draft narrative blocks (each becomes an HTML section): **Hero summary**, **Methods**, **Key findings**, **Per-candidate cards**, **Cross-cohort discussion** (if multi-cohort), **Limitations**, **Reproducibility**. Each receives the **full Fact Sheet** as a structured attachment with the directive: *"Copy every numeric value verbatim from the Fact Sheet. Do not infer, round, or compute new numbers. If a number you want is not in the Fact Sheet, output `[NOT IN FACT SHEET]` instead of inventing one. Use second-person voice for the hero, third-person for methods/limitations."*

Sections aim to satisfy the seven user-study quality dimensions (paper §6.1): **Novelty, Soundness, Presentation, Plausibility, Statistical Validity, Reproducibility, Limitations**. Limitations was the dimension where CoDaS scored highest (4.05 vs 1.69–2.14 for baselines) — explicit honest limitations are a feature, not a bug.

### E4. Quality gates (deterministic — apply before writing tables)
Suppress sections of the report when:
- **Multicollinearity gate**: VIF > 50 → suppress OLS coefficient tables; report Ridge only.
- **Performance gate**: best CV AUC < 0.55 OR CV R² < 0 → suppress ML benchmark tables, replace with "discriminative power below threshold".
- **Overfitting gate**: train/CV ratio > 5 → suppress feature importance.
- **Ablation gate**: all models at chance → suppress feature importance entirely.
- **Forest plot dedup**: max 2 representatives per feature family (e.g., only 2 of {sleep_duration_sd, sleep_duration_iqr, sleep_duration_mad}).
Each fired gate is recorded in `fact_sheet.json:quality_gate_flags` and shown as a transparent badge on the HTML page.

### E5. HTML report builder (deterministic)
Generate `<output_dir>/report.html` — **single self-contained file** (no external assets at runtime), Apple/Google production quality. Spec:

**Tech stack** (all CDN-loaded, then inlined for offline portability):
- **Typography**: `Inter` (UI), `JetBrains Mono` (numbers/code) via `@fontsource` inlined as base64. Fall back to system stack `-apple-system, BlinkMacSystemFont, "Segoe UI"` if inlining is not done.
- **Layout**: CSS Grid + Flexbox, max-width 1200px, generous whitespace (8px baseline grid).
- **Color system**: Light mode default, dark-mode toggle via `prefers-color-scheme`. Material Design 3 tonal palette OR Apple HIG semantic colors. Verdict badges: green (validated), amber (conditional), grey (rejected). One accent color per cohort.
- **Charts**: Chart.js (inline `<script>`, no CDN at view time):
  - Hero: animated Δ R² gauge (demographics-only → +biomarkers).
  - Forest plot: per-candidate ρ with 95% CI error bars, sorted by |ρ|, color-coded by verdict.
  - Validation battery heatmap: candidates × 11 tests, green/amber/red cells.
  - R² progression by round (line chart of best CV-R² across discovery rounds).
  - Subgroup consistency strip plot per top candidate.
  - Target distribution histogram with mean/SD overlay.
- **Motion**: subtle `IntersectionObserver` fade-in on scroll (200ms ease-out, 24px Y translate). No bouncing, no rotating, no parallax. Respect `prefers-reduced-motion`.
- **Navigation**: sticky top bar with anchor links to sections; collapsible per-candidate detail cards.
- **Mobile**: responsive at ≤768px; tables become scrollable cards.
- **Accessibility**: WCAG AA contrast, semantic HTML (`<section>`, `<article>`, `<nav>`), `aria-label` on charts, focus-visible outlines.

**Page structure** (top to bottom):
1. **Hero** — large title, dataset name, target, run timestamp. Three big stat cards: `n_validated`, `Δ R² over baseline`, `top candidate name + ρ`.
2. **Executive summary** — 3-sentence plain-language synthesis (LLM-written, copy-from-Fact-Sheet enforced).
3. **Top candidates** (cards, sorted by holistic score) — each card has: name, formula (mono), verdict badge, novelty tier (⭐ rating), Spearman ρ + 95% CI, sparkline of per-fold CV contribution, expandable detail (mechanism, all 11 tests, debate transcript snippet).
4. **Validation battery heatmap** — interactive: hover for per-cell test result, click to filter candidate list.
5. **Performance** — bar chart: demographics vs +biomarkers (regression: CV R²; classification: CV AUC), with per-fold scatter overlay.
6. **Discovery trajectory** — round-by-round Jaccard convergence + R² progression.
7. **Cohort profile** — demographics donut, target distribution histogram, missingness bar.
8. **Methods** — accordion with: pipeline diagram (inline SVG of the 6 phases), validation battery spec, leakage guards.
9. **Limitations** — softer background, prose; auto-include the standard CoDaS caveats (hypothesis-generating only, modest effects, no prospective validation).
10. **Reproducibility** — Fact Sheet hash, run config, seed, model versions, link to `audit.log`.
11. **Footer** — citation block (BibTeX collapsible), generation timestamp, "Generated by CoDaS skill — Claude Code".

**Implementation route:** write the HTML via a deterministic Python builder (`scripts/build_html.py`) that consumes `fact_sheet.json` + per-section LLM-drafted prose blobs and templates them into the page using f-strings (no Jinja runtime needed). Charts as `<canvas>` with inline JSON data tags. Keep CSS in a `<style>` tag at top, JS in a `<script>` at bottom. Validate the final HTML via `tidy -e` if available.

### E6. Numeric verification pass (deterministic)
Regex-scan the assembled HTML for numbers in: sample-size claims, candidate counts, test counts, R²/AUC values, ρ values. For each detected number, compare against Fact Sheet ground truth. If the LLM-emitted value is within 3× of truth and a key fuzzy-matches, auto-correct and log to `<output_dir>/phase_E/corrections.log`. Otherwise flag as `[VERIFICATION FAILED]` for human review.

### E7. Reviewer agent — final pre-publication QC (LLM)
Per paper Figure 7 ("Reviewer Agent"). Spawn a final LLM agent with the rendered HTML + Fact Sheet, asked to flag — *not fix* — any of:
- Hallucinated content (claims not derivable from Fact Sheet)
- Statistical errors (CIs not bounding point estimates, AUC < 0.5 reported as a finding, etc.)
- Logic errors (contradictory conclusions across sections)
- Biological contradictions (claims inconsistent with `literature_priors.json`)
Output `<output_dir>/phase_E/reviewer_flags.json`. If any high-severity flags, surface them to the user before declaring the run complete.

### E8. Final assembly
Output:
- `<output_dir>/report.html` — primary deliverable.
- `<output_dir>/summary.json` — machine-readable digest.
- `<output_dir>/report.md` — *only if `--with-paper` flag is set*. Markdown manuscript with the same content + figures embedded as base64 PNGs.

---

## Phase F — Human Feedback (optional, gated)

Before exiting, surface the report and pause for human input on three explicit triggers (per the paper):
- **Plausibility gap**: a high-effect candidate has no mechanism agent could ground in literature.
- **Conflicting evidence**: data-driven finding contradicts a literature_priors entry.
- **Search stagnation**: GapChecker exited at round 1 with < 3 validated candidates.

Restrict human input to: mechanistic interpretation, plausibility judgments, suggested feature transformations. **Never expose** raw target values or fold-level predictions to the human in this phase — that would breach leakage discipline (paper §2.5).

If the user provides guidance, optionally restart from Phase B with the new transformations injected into Hypothesis-agent context.

---

## Leakage-prevention invariants (NEVER violate)

These are non-negotiable. Hold yourself to them; if you cannot, abort and report.

1. **Raw variable exclusion** — `excluded_features` columns never enter the candidate pool.
2. **Transformation prohibition** — any candidate with |ρ| > 0.85 vs an excluded variable is dropped at Phase B4.
3. **Discovery-evaluation separation** — all CV uses `GroupKFold(n_splits=5)` keyed by participant ID. Hyperparameters tuned within training folds only (nested CV).
4. **Label isolation** — LLM agents see only summary statistics returned by Python runners (correlation magnitude/direction, p-value buckets). They never see raw target values, fold predictions, or participant-level rows.
5. **Construct overlap gating** — composites with high inter-feature correlation are reported once, not double-counted.

---

## Output contract

At the end of a successful run, produce:
- **`<output_dir>/report.html`** — primary deliverable. Single self-contained file, Apple/Google-quality interactive report. Open in any modern browser; no server needed.
- `<output_dir>/summary.json` — `{n_validated, n_conditional, top_candidates: [...], cv_metrics: {...}, run_duration_sec, fact_sheet_hash, reviewer_flags: [...]}`.
- `<output_dir>/phase_E/fact_sheet.json` — ground-truth numbers (the deterministic source `report.html` was templated from).
- `<output_dir>/phase_E/reviewer_flags.json` — final QC findings from Reviewer agent.
- `<output_dir>/phase_C/verdicts.json` — per-candidate per-test results.
- `<output_dir>/audit.log` — every agent invocation, every Python subprocess, every numeric correction.
- `<output_dir>/report.md` — *only if `--with-paper` flag is set*. Manuscript-format markdown for arXiv-style submission.

End the run with a ≤4-sentence text summary to the user: validated count, top-3 candidates by |ρ|, ΔR² over baseline, and the path to `report.html` (with a `file://` URL the user can open).

---

## Defaults & knobs

- Discovery rounds: `N=4` (paper used 4; Jaccard convergence at 0.80 typically exits at round 3)
- Permutation iterations: `1000`
- Bootstrap iterations: `1000`
- CV folds: `5`, `GroupKFold` keyed by participant ID
- Hard-gate threshold: `|ρ| > 0.85`
- FDR α: `0.05` (Benjamini–Hochberg, applied per-round across full univariate family)
- Minimum validated candidate effect size: `|ρ| ≥ 0.10`
- Discriminative AUC floor: `0.55`
- Composite share target: ≥40% of Hypothesis-agent proposals
- Imputation strategies for sensitivity check: `{median, knn-5, iterative}`
- Threshold sensitivity pair: default `{p<0.05, |ρ|≥0.20}` vs lenient `{p<0.10, |ρ|≥0.10}`

**Flags:**
- `--quick` — reduce permutation/bootstrap to 200, rounds to 2. Flag the run as draft; battery results less reliable.
- `--with-paper` — also emit `report.md` manuscript draft alongside the HTML report.
- `--no-literature` — skip Phase A3 Researcher ensemble (ablation: paper showed −Lit reduces interpretability but not raw R²).
- `--seed=<int>` — fix RNG seed for reproducibility (default: 42).

---

## Things to NOT do

- Do not hand-write candidate biomarkers from training-data memorization of the paper. The pipeline's value is autonomous discovery; reproducing the paper's named biomarkers from memory defeats the purpose.
- Do not let any LLM agent emit final numerical values for the report — always copy from Fact Sheet.
- Do not skip the participant-level GroupKFold; per-row splits will leak in repeated-measures cohorts.
- Do not run Critic and Defender sequentially — they must be parallel agents in a single message so neither sees the other's argument before forming its own.
- Do not print the user's raw target values in the report.

---

## Reference

Kim, Y., Rahman, S., Schmidgall, S., Park, C., Heydari, A.A., Metwally, A.A., Yu, H., Liu, X., Xu, X., Yang, Y., Xu, M.A., Zhang, Z., Breazeal, C., Althoff, T., Sirkovic, P., Rendulic, I., Pawlosky, A., Stroppa, N., Gottweis, J., Vedadi, E., Karthikesalingam, A., Kohli, P., Natarajan, V., Malhotra, M., Patel, S., Park, H.W., Palangi, H., McDuff, D. (2026). *CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors.* arXiv:2604.14615.

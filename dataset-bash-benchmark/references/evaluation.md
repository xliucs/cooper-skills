# Frozen experiments, scoring and evidence

Read this before launching scored inference and when generating the report.

## Record the experiment contract

Identify which upstream tasks, prompts, CLI/environment, default budgets and graders are reused verbatim, and list every intentional deviation. Audit both upstream restrictions and adapter tool routing.

The frozen manifest must identify:

- Dataset version/checksums, split units, task IDs, construction seeds and any task admission rule.
- Exact public-file set and hashes; private answers and grader hashes stored separately.
- Task and runtime prompts, tool schemas, package lock/image digest, client and harness versions.
- Requested and resolved model identifiers, identity evidence level, authentication/billing mode.
- Reasoning/effort settings, tool/image capabilities, per-command/episode deadlines, actions, output/artifact limits, concurrency and attempts per task.
- Task order/randomization, primary metrics, tolerances, controls, baseline scores, exclusions, inference retry policy and analysis plan.

Use one attempt per task/model for a small initial pilot unless repetitions were requested or explicitly included in the stated budget. For 50 tasks and three models, that is 150 scheduled attempts, not automatically 450. Canary/development runs are separate and counted separately in usage.

Freeze after development checks and before scored runs. Hashes must be checked on run/resume/regrade. Changing the prompt, representation, budget or metric creates a new version. Rerun all conditions needed for that new comparison; preserve the original results. A hinted parameter-estimation task and a blind artifact-repair task do not form a pure prompt ablation.

## Baselines and identifiability

Report a no-op/trivial baseline, a reference solver with its information access disclosed, and unchanged controls where relevant. Check absolute success and improvement over input separately. Privileged oracle success validates the transformation/grader pipeline; it does not demonstrate that the public observations uniquely identify a solution.

When blind agents fail but the oracle passes, inspect the actual trajectories and the information available to the actor. A strict reconstruction miss can reflect a different reasonable interpretation, insufficient time, numerical tolerance, or an unidentifiable problem. Distinguish these possibilities before diagnosing lack of reasoning. Human-like exploration is a design goal, not evidence of human-equivalent performance without a human baseline.

A high no-op score is not automatically a broken benchmark: expected clean controls should pass. Report corrupted and control subsets separately. If corrupted no-op outputs pass too often or the metric admits a shortcut, document the flaw and create a new development-tested version; never tighten the completed run's thresholds to manufacture separation.

If a reference/admission rule filters candidate tasks, report candidate, admitted and rejected counts plus the rule. Do not present the reference's pass rate on its own admitted set as independent held-out performance. Include intrinsically ambiguous cases only with an observable, fair success criterion such as calibrated abstention.

## Comparisons and budgets

Run paired task IDs with matched public information, outputs, tools, action/time ceilings and repetitions. Interleave or randomize model order to reduce time-of-day/capacity confounding. Maintain fresh conversations/workspaces between episodes. Agent scaffolds, default prompts, context limits and reasoning controls can differ across providers: report them and describe cross-provider results as **model-plus-harness comparisons**.

Separate model-tier experiments from test-time scaling. For the latter, hold model, tasks, prompts and grading fixed while varying one declared resource budget. Identically named “medium” settings do not guarantee equal inference compute across vendors. Record token counts and cache use where available, and wall time/provider latency. Do not infer parameter scaling or a scaling law from unknown model sizes and three tier labels.

## Failure and retry accounting

Maintain an append-only attempt ledger keyed by experiment/model/task/repetition/attempt. Useful terminal states include valid submission, invalid submission, no submission, deadline/action exhaustion, protocol violation, provider capacity, rate/quota limit, auth failure, model unavailable/mismatch and harness failure. Preserve raw error evidence, timestamps, partial usage and artifacts without secrets.

Save the upstream artifact reward separately from any operational penalty for protocol or infrastructure failure. Specify allowed native discovery/bookkeeping calls before the run. If an accepted answer receives an operational zero because of an extra client call, preserve the call's returned information and explain the distinction. A secondary artifact-score analysis can isolate harmless interface failures only after checking that the extra calls provided no forbidden information or other advantage; contaminated answers are not clean capability evidence. Label any analysis added after seeing failures as post-hoc, and never change the frozen primary accounting retroactively.

Default to no replacement of interrupted scored attempts. A process may have consumed a model completion even if no final result was written. Resume completed records by verifying their hashes; treat a partial attempt as interrupted, not as an unused slot. An accepted submission remains terminal even if later client shutdown reports an error.

If infrastructure retries are permitted, predeclare retryable statuses, maximum attempts, backoff, treatment of uncertain request completion, and how the primary metric uses first attempts versus recovered attempts. Preserve all attempts, label recovery results separately, and never pick the best answer. If the user authorizes retries only after seeing results, report them as a new/post-hoc recovery analysis.

Publish both coverage and outcomes. A complete scheduled-attempt table may count infrastructure failures as operational non-successes, but label it accordingly and never call capacity failures evidence of model inability. Also report a paired sensitivity analysis on common executable task IDs, excluding the same affected IDs from every model. Retain ordinary model timeouts and invalid repairs. Mark this subset as secondary; it does not remove selection bias. For unstarted quota-blocked work, show scheduled, started and completed counts rather than inventing missing rollouts.

## Statistics and reporting

Report the upstream primary metric per model, task/control/subgroup breakdowns, execution status counts and time/actions/tokens. Report strict success only when the task defines it; do not invent a pass threshold for continuous rewards. Include image usage only when applicable. Distinguish preserved input, changed output and no valid artifact for repair tasks. Diagnostic prose is not an automatic label of correct understanding.

Use paired uncertainty, such as bootstrap differences at the independent sampling unit and exact paired tests for binary outcomes. With repeated trials or multiple windows per subject, cluster appropriately; do not treat them as independent subjects. Specify strata, seeds and repetitions; adjust multiple planned pairwise tests when appropriate. A one-rollout pilot cannot estimate same-task stochastic variance reliably.

Include examples of success, failure, no-op and model disagreement. State post-hoc case-selection rules. Show the actual artifacts, shared axes/units, and any grading equivalences consistently. If the grader uses a common time registration, apply the same rule to both original and repaired rows so an unchanged output cannot appear improved. Baseline/script/oracle examples must not be presented as model output.

Audit claimed visual use against actual image-capable observations and available client evidence. Autonomous plot use is observational; estimating the benefit of vision requires a controlled tool-availability experiment. Preserve sanitized observable actions and final explanations, not private chain-of-thought or authentication material.

Before delivery, verify task/model grid completeness and duplicates; regrade trusted snapshots; check identity and image claims at the evidence level available; run environment regression tests; execute requested notebooks in a fresh workspace; inspect plots and report navigation; and verify the shared URL. Report uncovered capability/portability limits explicitly.

## Files to deliver

Keep source and small manifests in the requested repository. Keep newly created benchmark repos private by default. For large data or trajectory archives, use authorized private artifact storage/release assets with hashes and retrieval instructions instead of bloating Git history. Honor dataset redistribution limits. Include:

1. Acquisition and inspection summary, concrete environment plan, upstream task exporter or necessary builder, adapter changes and package/runtime lock.
2. Public task packages, separately protected private verifier, and tested multi-turn runner or RL `reset`/`step` interface.
3. Frozen experiment manifest, complete attempt ledger, model identity evidence, submitted-artifact hashes and regrade results.
4. HTML report and machine-readable tables; a runnable notebook when requested, clearly labeled inference versus replay.
5. Exact reproduction commands, unavailable conditions, and a concise statement of what the experiment establishes and what remains untested.

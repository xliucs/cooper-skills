# Subscription-backed Codex and Claude Code runners

Read only the selected provider's section plus the shared preflight. Documentation and locally installed CLI help were checked on **2026-10-08**; recheck on each new implementation because flags, model access and plan behavior change.

## Shared preflight and authentication boundary

Use official native clients through their supported unattended modes. Let the client manage subscription authentication; do not extract OAuth tokens into a generic API client, rewrite private provider traffic, spoof client identity, rotate accounts to evade limits, or promise unlimited plan usage.

Record client version, supported flags, active authentication mode, account-visible model options, model configuration and identity evidence. Inspect configuration and the *presence* of authentication/provider environment variables without logging their values. Subscription-only means no implicit API/cloud-provider fallback, paid extra usage, credits purchase or reset redemption. If billing mode cannot be verified, finish offline work and request the needed login/decision before inference.

Keep changes to the experiment's child-process environment/configuration. Do not log out the user's normal client or overwrite global settings. Unsetting an API-key variable alone is insufficient: inspect active profiles, key helpers, custom endpoints and cloud-provider settings as well. Avoid loading unrelated skills, memories, hooks, MCP servers or personal AGENTS/CLAUDE instructions into an evaluated actor. Retain necessary account authentication using supported client mechanisms and audit the effective context.

The native client runs on the trusted side of the environment boundary described in [environment.md](environment.md). A headless CLI command alone does not implement task isolation, budgets, terminal submission, or an optional image bridge. If using Harbor, inspect its actual agent implementation and auth requirements first: its [published integration example](https://docs.harborframework.com/agents/pre-integrated-agents) uses an API key. Do not assume the installed adapter supports subscriptions because it invokes the same CLI.

## Codex

Check `codex --version`, `codex exec --help` and `codex login status`. Use an existing verified **Sign in with ChatGPT** session, or have the user complete the official login flow. API-key authentication is a separate billing path. Sources: [authentication](https://learn.chatgpt.com/docs/auth), [non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode).

Prefer `codex exec` with machine-readable events, or a documented app-server/SDK interface when its current authentication and tool routing meet the contract. Resolve the user's exact models from supported account-visible metadata, then test each. The example labels Luna/Terra/Sol are not a model-discovery API, nor permission to choose a replacement.

A launch *shape*, after configuring and testing the environment boundary:

```bash
codex exec --model "$MODEL_ID" --json \
  --output-last-message "$RUN_DIR/final.txt" \
  - < "$PROMPT_FILE" > "$RUN_DIR/events.jsonl" 2> "$RUN_DIR/stderr.log"
```

Choose explicit supported sandbox, approval, tool-server and configuration settings for the backend. A task needs permitted writes; default read-only execution may not suffice. Workspace-write does not by itself prove that only task files are readable. Use `--skip-git-repo-check` only when needed for the controlled task workspace. Configuration-isolation flags do not automatically prove personal instructions and all native tools are absent: test the effective setup.

Capture whatever model identity the official client exposes, with its provenance. Requested `--model` and the model's self-reported name are not independent attestation. If only configured identity is observable, mark it as such; do not intercept private endpoints merely to strengthen the claim.

## Claude Code

Check `claude --version`, `claude --help` and the supported auth-status command. Verify a claude.ai subscription session, not a Console/API/cloud-provider login. Confirm current entitlement for each requested exact model. Sources: [authentication](https://code.claude.com/docs/en/authentication), [Pro/Max access](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan).

Use native `claude -p` for unattended runs and stream JSON for tool trajectories. Sources: [programmatic execution](https://code.claude.com/docs/en/headless), [CLI reference](https://code.claude.com/docs/en/cli-reference).

```bash
claude -p --model "$MODEL_ID" --output-format stream-json --verbose \
  < "$PROMPT_FILE" > "$RUN_DIR/events.jsonl" 2> "$RUN_DIR/stderr.log"
```

Configure an explicit tool allowlist/permission policy and the tested sandbox bridge. Distinguish the set of available tools from tools preapproved to run. Do not set automatic fallback models in an identity-controlled experiment. Preserve stream results, errors, tool inputs/outputs and per-model usage metadata; an estimated USD field is not proof of an actual API charge or of free subscription usage.

**Version-specific trap:** locally checked Claude Code 2.1.294 documents that `--bare` skips OAuth/keychain authentication. Do not copy the general CI recommendation to a subscription runner without checking the current behavior. Isolate context using supported controls that retain subscription login, and verify with a real canary. Do not silently repair an auth failure by injecting an API key.

Do not assume the separately distributed Agent SDK accepts subscription authentication on the same terms as the native CLI; verify its current documentation before choosing it. Compare available Claude tiers only after resolving account access and exact identifiers. Record alias resolution and any version change during a long run.

## Canary, limits and recovery

For each model, execute a disposable task that uses a small development task through the actual upstream CLI/API, performs at least two dependent tool actions, and submits in the upstream format. Include plot creation and pixel delivery only if the chosen environment exposes visual tools. Verify the graded artifact, tool routing, model evidence, image-delivery evidence when applicable, deadline handling and cleanup. This canary must not reveal scored task solutions or count toward their denominator.

Start with one or a few workers; increase only within the plan's limits. Account for cumulative shared subscription usage. On capacity/rate/auth errors, save the failed attempt and stop or back off new dispatch according to the frozen policy. Do not switch accounts, models or billing modes to keep a graph moving.

Preflight-unavailable models remain unavailable. If a model alias or provider identity changes mid-run, preserve affected results, stop that condition, and report the version break. Keep healthy independent work moving without representing a partial comparison as complete. See [evaluation.md](evaluation.md) for retry and denominator handling.

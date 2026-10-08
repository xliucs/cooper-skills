# Terminal environment and task contract

Read this when implementing the environment, tools, task packaging, and grader.

## Boundary between builder, actor and grader

Use three distinct contexts:

- **Builder/controller:** downloads data, selects splits, creates tasks, starts agents and records outcomes. It may see all sources and labels.
- **Actor workspace:** one task's permitted input, schema, task text and writable scratch/output. It must not contain builder notes, evaluation reports, labels, other tasks, the source repository or the skill that built the benchmark.
- **Trusted grader:** receives a frozen copy of the submitted artifact after actor execution ends. It alone receives private answers and scoring code. Return terminal reward to the trainer; do not feed it back into a scored rollout.

Use an actual process/filesystem boundary. A `private/` sibling folder, a working-directory flag, a hidden tool description or a prompt saying “do not read answers” is not isolation. Protect container build context and image layers too: deleting truth files from a later layer is insufficient. Do not copy the whole benchmark repository into the actor image.

Prefer an isolated container/VM per episode, bounded CPU/RAM/disk/process counts, and no actor network after provisioning unless the task explicitly requires it. Keep the authenticated model client on the trusted side with provider connectivity; route its execution/image tools to the task sandbox. The actor shell should have neither the client credentials nor unrestricted host tools. If a native sandbox is the only practical backend, test its boundaries and disclose its limits.

For a shared tool bridge, provide Bash execution, image reading and submission through supported MCP or native integration points. Native shell, file-read, browser, search, and delegation tools must not offer a route around that boundary. Disable unneeded capabilities with supported controls and audit unexpected calls. If enforceable restriction is unavailable, label the setup cooperative; do not claim a hardened sandbox.

## Packages and state

A useful CPU starting point is Bash, coreutils, `find`, `sed`, `awk`, `jq`, `ripgrep`, Python, NumPy, pandas, SciPy, matplotlib, Pillow and pytest. Add pyarrow/DuckDB, scikit-learn, h5py, soundfile, ffmpeg or domain readers only when needed. Download tools belong in provisioning when the actor does not need network access. Pin the base image digest, Python version and resolved dependency versions; record the installed inventory.

Keep filesystem changes throughout an episode. Define shell semantics explicitly: separate Bash calls need not preserve exported variables or `cd`; provide a fixed workdir or a real persistent shell and test the chosen behavior. A fresh episode resets both files and conversation state. Seed generation/splits independently of any model sampling control; do not imply that a CLI seed makes model outputs deterministic.

## Multi-turn action loop

Expose a small, provider-neutral contract, adapted to the chosen framework:

```python
observation, info = env.reset(task_id=task_id)
while True:
    action = policy.next(observation)  # retain the episode's conversation
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

| Action | Observation / effect |
|---|---|
| `bash(command, timeout)` | Execute in the same task filesystem; return exit status, bounded stdout/stderr, and timeout status. |
| `view_image(path)` | Validate a workspace image, decode it, and deliver image pixels through the client's actual image-capable tool path. |
| `submit(artifact_path, report)` | End the episode once; controller snapshots and validates the artifact, then runs the private grader. |

Saving a PNG or returning its filename/base64 as plain text is not evidence that the model saw it. Preserve the actual image payload or a trustworthy client event with its hash and delivery status. Distinguish created, tool-returned and model-consumed images; label consumption unverified if the client cannot attest it.

Enforce total actions, per-command wall time, overall wall time, output length and artifact size in the controller, not only in the prompt. Count each tool action inside batches; define whether submission consumes a slot. Model turns, Bash actions, parallel calls and tokens are different quantities. A client `max_turns` flag is not a shell-action limit.

On submission or timeout, prevent further actions, terminate the actor and its child processes, and atomically snapshot the final artifact. Reject duplicate/post-terminal submission. Resume bookkeeping from committed terminal records, not from a half-written result.

## Verifiable output

Use deterministic artifact grading when the objective allows it. Check paths, symlinks/nonregular files, size limits, archive contents, dtype, shape, units and finite values before processing untrusted artifacts. Avoid unsafe pickle loading. Bound parser and grader resources. Validate the controller's artifact snapshot on later regrading instead of rereading mutable actor files.

Separate format validity, diagnostic correctness (if measured), repair/solution quality and unnecessary changes. Free-text explanations are evidence for inspection, not automatically correct diagnoses. If a judge is required, freeze its rubric/version and validate it against labeled examples; do not let the tested model grade itself.

For reconstruction, score the observable requested result and accept valid equivalent representations. For example, if neither channel was declared a trusted clock, a common change of time reference can be legitimate. Do not independently align each channel in a way that erases the error being tested. Test constants, smoothing, dropping hard samples, copying inputs and malicious artifacts as potential reward shortcuts.

## Harbor packaging

Use the [current Harbor task format](https://docs.harborframework.com/tasks/overview) and the installed version's schema. Typical components are:

```text
task-id/
  instruction.md
  task.toml
  environment/Dockerfile
  solution/solve.sh
  tests/test.sh
```

Only actor-permitted inputs belong in the environment build. Solution and verifier resources are for the controller's appropriate execution stage. Follow the selected Harbor version's verification/reward-file contract; confirm the tests and solution are unavailable during actor execution. Validate with an oracle and a no-op run through Harbor itself before claiming compatibility.

For a custom runner, separate `acquire`, `build_tasks`, `preflight`, `run`, `regrade` and `report` entry points. Retain a portable task manifest so Harbor export is straightforward. Do not claim that a macOS-only launcher also runs inference in Linux or Colab.

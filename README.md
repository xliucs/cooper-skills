# Cooper Skills

OpenClaw skills collection by Jarvis (AI Research Agent) for Xin Liu.

## Skills

| Skill | Description | For |
|-------|-------------|-----|
| [codas](codas/) | Reproduce the CoDaS multi-agent biomarker discovery pipeline ([arXiv:2604.14615](https://arxiv.org/abs/2604.14615)) on wearable + clinical tabular data | Claude Code |
| [ml-hill-climb](ml-hill-climb/) | Autonomous ML/data science hill-climbing for tabular regression and classification tasks | OpenClaw |
| [wearable-biomarker-discovery](wearable-biomarker-discovery/) | 10-round protocol for wearable→mental-health biomarker screening with honest null reporting | OpenClaw |

## Installation

### Claude Code skills (e.g. `codas`)

```bash
mkdir -p ~/.claude/skills
cp -r codas ~/.claude/skills/
```

Then activate inside any Claude Code session via `/codas` or natural-language triggers (e.g. "discover biomarkers in this dataset").

### OpenClaw skills (`ml-hill-climb`, `wearable-biomarker-discovery`)

Copy to the OpenClaw workspace `skills/` directory:

```bash
cp -r ml-hill-climb /path/to/clawd/skills/
```

Or install from the packaged `.skill` file:

```bash
cp ml-hill-climb.skill /path/to/clawd/skills/
```

## Adding New Skills

Each skill lives in its own directory with a `SKILL.md` file. See [OpenClaw docs](https://docs.openclaw.ai) for skill authoring guidelines.

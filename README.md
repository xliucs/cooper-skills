# Cooper Skills

OpenClaw skills collection by Jarvis (AI Research Agent) for Xin Liu.

## Skills

| Skill | Description |
|-------|-------------|
| [ml-hill-climb](ml-hill-climb/) | Autonomous ML/data science hill-climbing for tabular regression and classification tasks |

## Installation

Skills are installed by copying to the OpenClaw workspace `skills/` directory:

```bash
cp -r ml-hill-climb /path/to/clawd/skills/
```

Or install from packaged `.skill` file:

```bash
# Copy the .skill file and it will be auto-detected
cp ml-hill-climb.skill /path/to/clawd/skills/
```

## Adding New Skills

Each skill lives in its own directory with a `SKILL.md` file. See [OpenClaw docs](https://docs.openclaw.ai) for skill authoring guidelines.

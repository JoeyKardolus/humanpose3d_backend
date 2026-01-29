# Documentation

## Quick Reference

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | **Start here** - Quick start, usage, common issues |
| [CLAUDE.md](../CLAUDE.md) | Contributor and AI assistant guidelines |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagram, data flow, module responsibilities |

## Technical Documentation

| Document | Purpose |
|----------|---------|
| [NEURAL_MODELS.md](NEURAL_MODELS.md) | Neural refinement models (POF + joint + MainRefiner) |
| [POF_EXPLANATION.md](POF_EXPLANATION.md) | Part Orientation Fields math and concepts |
| [OUTPUT_ORGANIZATION.md](OUTPUT_ORGANIZATION.md) | Output file structure and naming |

## Development

| Document | Purpose |
|----------|---------|
| [CHANGELOG.md](CHANGELOG.md) | Development history and milestones |
| [AGENTS.md](../AGENTS.md) | Development guidelines and code style |

## Archive

Historical documentation preserved in `archive/`:

```
archive/
├── sessions/           # Development session logs
├── fixes/              # Bug fixes and improvements
├── decisions/          # Technical decision documents
├── deprecated/         # Obsolete feature docs
└── BUILD_LOG_runs.md   # Verbose pipeline run history
```

---

*For the most up-to-date CLI options, run: `uv run python manage.py run_pipeline --help`*

Storage note: non-code assets are stored under `~/.humanpose3d` by default (override with `HUMANPOSE3D_HOME`).

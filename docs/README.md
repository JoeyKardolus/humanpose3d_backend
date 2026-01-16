# Documentation

## Quick Reference

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](../CLAUDE.md) | **Start here** - Quick start, usage, common issues |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagram, data flow, module responsibilities |
| [CLAUDE_EXTENDED.md](CLAUDE_EXTENDED.md) | Detailed reference for all features |

## Current Development

| Document | Purpose |
|----------|---------|
| [NEURAL_MODELS.md](NEURAL_MODELS.md) | Neural refinement progress (depth + joint constraints) |
| [CHANGELOG.md](CHANGELOG.md) | Development history and milestones |

## Feature Documentation

| Document | Purpose |
|----------|---------|
| [MULTI_CONSTRAINT_OPTIMIZATION.md](MULTI_CONSTRAINT_OPTIMIZATION.md) | 3-phase optimization pipeline |
| [OUTPUT_ORGANIZATION.md](OUTPUT_ORGANIZATION.md) | Output file structure and naming |

## Setup Guides

| Document | Purpose |
|----------|---------|
| [VIDEOPOSE3D_SETUP.md](VIDEOPOSE3D_SETUP.md) | VideoPose3D integration (future) |
| [FLK_SETUP.md](FLK_SETUP.md) | FLK filter setup (deprecated) |

## Archive

Historical documentation preserved in `archive/`:

```
archive/
├── sessions/           # Development session logs
├── fixes/              # Bug fixes and improvements
├── decisions/          # Technical decision documents
└── BUILD_LOG_runs.md   # Verbose pipeline run history
```

## Documentation Method

### Active Docs (root level)
- **Reference docs**: Architecture, extended reference, usage
- **Progress trackers**: Neural models, changelog
- **Feature guides**: Specific features still in use

### Archived Docs
- **Session logs**: Auto-generated session reports
- **One-time reports**: Bug fixes, cleanup reports, testing reports
- **Decisions**: Technical decision documents (VideoPose3D, etc.)
- **Deprecated**: Features no longer in active use

### Naming Conventions
- `FEATURE_NAME.md` - Active feature documentation
- `archive/fixes/feature-name.md` - Historical fix documentation
- `archive/sessions/YYYY-MM-DD_description.md` - Session logs

### When to Archive
- Session reports after session ends
- Bug/fix docs after issue resolved
- Decision docs after decision made
- Feature docs when feature deprecated

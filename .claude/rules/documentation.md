# Documentation Rules

## Automatic Documentation Updates

After completing significant work (new features, bug fixes, architecture changes), proactively:

1. **Update CLAUDE.md** if commands, workflows, or architecture changed
2. **Update relevant docs/** files if user-facing behavior changed
3. **Add to CHANGES.md** for notable changes

## When to Update Docs

- New CLI flags or options added → Update CLAUDE.md "Important Flags" section
- New module or significant refactor → Update "Module Responsibilities" section
- Bug fixes affecting usage → Add to "Known Constraints & Troubleshooting"
- New workflow discovered → Add to "Recommended Workflow" section

## Session Tracking

Before ending a session or when context is getting long:
- Summarize what was accomplished
- Note any pending work or issues discovered
- Update PROGRESS.md with current status

## File Naming

- Session reports: `docs/sessions/YYYY-MM-DD_HHMM.md`
- Keep CLAUDE.md as the canonical reference
- Use PROGRESS.md for ongoing work tracking

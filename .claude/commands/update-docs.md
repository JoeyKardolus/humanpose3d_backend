---
description: Update project documentation with recent changes
---

Review the recent git changes and update documentation:

1. Check recent commits:
!`git log --oneline -15`

2. Check changed files:
!`git diff --name-only HEAD~10`

3. Update @CLAUDE.md if any of these changed:
   - CLI flags or options
   - Module structure
   - Workflows or commands
   - Architecture

4. Update @PROGRESS.md with current status

5. If there are breaking changes, note them in @CHANGES.md

Be concise - only document what actually changed.

# AI Assistant and Contributor Guidelines

Instructions for anyone working in this repository (humans and AI assistants).

## Architecture Principles

### General
- Keep it simple and readable.
- Single Responsibility Principle everywhere.
- Prefer explicit code over clever code.
- Many small files are better than a few large ones.
- If something feels too convenient, double-check it.

### Front-end

Markup
- Use plain HTML.
- No inline JavaScript.
- No inline CSS (except dynamic values such as `style="width: {{ value }}%"`).
- No `<script>` tags inside HTML files.

Styling
- Primary framework: Bootstrap.
- Custom styling: plain CSS in dedicated `.css` files under `src/application/static/`.
- No Tailwind, no inline styles, no `<style>` blocks.
- Prefer Bootstrap utilities before adding custom CSS.

Front-end logic
- Use JavaScript (ES6+) in dedicated `.js` files under `src/application/static/`.
- No inline JavaScript.
- HTML may only reference bundled or compiled assets.

Business logic never belongs in the front-end.

### Back-end

Architecture
- Backend code follows strict Object-Oriented Programming.
- Use Django MVT with app-based domain structure.
- Think in apps as domains, not layers.

Django rules
- Do not overload framework files (`views.py`, `models.py`, `signals.py`).
- Split logic into dedicated modules:
  - `controllers/` for thin entry points
  - `services/` for business logic
  - `use_cases/` for orchestration
  - `repositories/` for data access
  - `validators/` for validation logic
  - `dto/` for data transfer objects
  - `config/` for configuration

Application boundary
- All application logic lives in `src/application/`.
- No business logic outside `src/application/`.
- No side effects leaking into framework or infrastructure layers.

Framework = shell. Application = core.

### Database policy
- The Django app is configured to use the dummy database backend.
- Do not introduce database dependencies unless explicitly requested.

### Storage policy
- Store non-code assets under `~/.humanpose3d` by default.
- Respect `HUMANPOSE3D_HOME` if set.

## Coding Style

- Follow PEP 8.
- Use descriptive `snake_case`.
- Type-hint all public interfaces.
- Add docstrings to modules, classes, and public methods.
- Use inline comments only when logic is non-obvious.
- No wildcard imports.
- No circular dependencies.

## Testing

- Use `pytest`.
- Test files mirror `src/` structure.
- Prefer deterministic fixtures.
- Mock heavy OpenCV and MediaPipe dependencies.
- Document new test data in PRs.

## Commit and PR Guidelines

- Use imperative, scoped commit messages.
- Keep unrelated changes separate.
- PRs require a short summary and validation notes.
- CI must be green before merge.

## Final Notes

- Avoid framework shortcuts.
- Make unclear things explicit.
- Readability beats cleverness.

Explicit > Implicit
Structure > Speed

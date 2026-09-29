# kaggle-tools

Python utilities (`src/kaggle_tools/`) + a Claude Code plugin of Kaggle skills (`skills/`).

## Commands

```bash
uv sync
uv run pytest
uv run ruff format . && uv run ruff check .
uv run ty check src
claude plugin validate .              # after editing .claude-plugin/ or skills/
```

## Conventions

- Core deps stay `numpy`, `pandas`, `scikit-learn`. Anything heavier goes in an optional
  extra and is imported lazily inside the function that needs it.
- Modules are independent and small; each module docstring states the competition lesson
  it encodes. Keep that link when adding code: utilities exist because something failed.
- Every public function gets a test in `tests/`.
- Skills: one `SKILL.md` per folder with `name`/`description` frontmatter. Record
  evidence (competition, version, LB delta) for every "works"/"doesn't work" claim,
  and reference the matching `kaggle_tools` helper where one exists.
- Bump the version in `pyproject.toml` and both `.claude-plugin/*.json` together.

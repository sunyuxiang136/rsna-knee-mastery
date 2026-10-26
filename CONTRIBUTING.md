# Contributing

Thanks for helping! This project is small and focused: tested building blocks plus annotated reference
code for a public RSNA knee MRI pipeline.

## Setup

```bash
git clone https://github.com/sunyuxiang136/rsna-knee-mastery.git
cd rsna-knee-mastery
python -m venv .venv && source .venv/bin/activate
make install
```

## Workflow

1. Open an issue first for anything larger than a typo.
2. Branch from `main`: `git switch -c fix/short-description`.
3. Make focused changes. Add or update tests in `tests/` for any change to `src/`.
4. Run `make lint test demo` — CI runs exactly these.
5. Open a PR using the template. Keep commits in [Conventional Commits](https://www.conventionalcommits.org/) style
   (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

## Ground rules

- `src/rsna_knee` stays NumPy-only so tests run without a GPU or datasets.
- Behaviour that mirrors the reference code must cite where (file/constant) in a docstring or comment.
- Do **not** add patient data, DICOMs, competition data, model weights or credentials. Never commit tokens.
- Files in `reference/` are upstream code; change them only to fix extraction errors, and keep their notices.
- Code is MIT-licensed (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exceptions).

## Reporting security issues

See [SECURITY.md](SECURITY.md).

## Troubleshooting

**`ModuleNotFoundError: No module named 'rsna_knee'` on macOS after `pip install -e .`** — with Python 3.13+,
an editable install can be ignored if macOS marks the `.pth` file in your venv as hidden. Fix:
`chflags nohidden .venv/lib/python3.*/site-packages/*.pth` (or recreate the venv outside synced folders).
`pytest` is unaffected because `pyproject.toml` sets `pythonpath = ["src"]`.

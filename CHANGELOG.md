# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-26

### Added
- `rsna_knee` package: physical-size cropping, fixed-slot slice sampling, per-finding attention
  pooling and Stage 3 percentile-rank fusion with failure handling (NumPy only).
- Unit tests (pytest) and ruff lint config; GitHub Actions CI on Python 3.9–3.12.
- Reference code and Kaggle notebook under `reference/` and `notebooks/`.
- Zero-GPU teaching demo in `examples/beginner_demo.py`.
- Docs: architecture, reproduction guide, third-party notices.
- Community files: contributing, code of conduct, security policy, issue/PR templates.

# Repository Guidelines

## Project Structure & Module Organization
This repository is a design artifact plus a partial Python package. Key locations:
- `vaerans_ecs/`: source package. `core/` (arena, pipeline, world), `components/` (data types), `systems/` (transforms), `eval/`, `viz/`, and `api.py`.
- `tests/`: pytest tests.
- `examples/`, `benchmarks/`: usage and performance experiments.
- `models/`: expected ONNX model assets (avoid committing large binaries).
- `diagrams/`, `SOFTWARE_DESIGN.md`, `IMPLEMENTATION_STATUS.md`: architecture docs and status notes.
- Config: `vaerans_ecs.toml.example` (copy to `vaerans_ecs.toml`).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` installs editable package with dev tooling.
- `pytest` runs the test suite.
- `pytest --cov=vaerans_ecs --cov-report=html` generates coverage in `htmlcov/`.
- `mypy vaerans_ecs` runs strict type checks.
- `ruff check vaerans_ecs` lints; `black vaerans_ecs` formats (line length 100).

## Coding Style & Naming Conventions
Use 4-space indentation and Python 3.9+ features. Follow `black` formatting and `ruff` linting; keep modules in snake_case (for example, `components/latent.py`). Type hints are expected and enforced with `mypy --strict` and the Pydantic plugin. Test files follow `tests/test_*.py`, with `Test*` classes and `test_*` functions.

## Testing Guidelines
Write pytest unit tests alongside any new system or component behavior. Prefer small, deterministic tests. Run a focused test with `pytest tests/test_<module>.py -v` before broader suite runs. When validating VAE behavior, test against the real SDXL VAE ONNX models (configured in `vaerans_ecs.toml` or `VAERANS_CONFIG`), not mock or stub models.

## Commit & Pull Request Guidelines
No Git history is included in this artifact, so there is no established commit message convention to summarize. Use a short, imperative subject (for example, `Add wavelet quantizer checks`) and keep related changes together. For PRs, include a clear description, link any relevant issues, note test coverage, and update design docs if behavior changes. Avoid committing local model binaries or large generated files.

## Configuration & Agent Notes
Runtime model paths live in `vaerans_ecs.toml` (or `VAERANS_CONFIG`). Do not store secrets in configs. When making architectural changes, align with `CLAUDE.md` and `SOFTWARE_DESIGN.md` to preserve the ECS layering and zero-copy design intent.

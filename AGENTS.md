# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `src/myvllm/`. Use `engine/` for runtime pieces such as the scheduler, sequence state, block manager, and model runner; `layers/` for attention, normalization, linear, and sampling primitives; `models/` for model definitions such as `llama.py` and `qwen3.py`; and `utils/` for shared helpers. Root scripts are task-oriented entrypoints: `main.py` and `main_llama32.py` run demos, while `benchmark_prefilling.py`, `benchmark_decoding.py`, and `benchmark_tps.py` measure performance. Tests currently live in `tests/`, with scheduler notes in `tests/scheduler_tests.md`. Treat `src/myvllm.egg-info/` as generated output, not source.

## Build, Test, and Development Commands
Use `uv` first because `uv.lock` is committed.

- `uv sync` installs pinned runtime and dev dependencies.
- `uv run python main.py` runs the main inference demo.
- `uv run python benchmark_prefilling.py` benchmarks prefill attention paths.
- `uv run python benchmark_decoding.py` benchmarks decode-time page attention.
- `uv run pytest tests/test_scheduler.py -v` runs the current unit test suite.
- `uv run black src tests` and `uv run isort src tests` format code and imports.

If `uv` is unavailable, use `python -m pip install -e ".[dev]"`.

## Coding Style & Naming Conventions
Target Python 3.11. Use 4-space indentation, type hints on public functions, and small, single-purpose methods. Follow existing naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; short descriptive benchmark script names like `benchmark_decoding.py`. Keep comments sparse and explain non-obvious scheduling or caching logic, not basic syntax.

## Testing Guidelines
Write tests with `pytest` and name them `tests/test_<feature>.py`. Match the current style: isolate scheduler or cache invariants with focused regression tests and use `MagicMock` for block-manager behavior when possible. For changes in scheduling, paging, or caching, add at least one failure-driven test that proves no sequence or block state is lost. If GPU-only behavior cannot be unit-tested, include the benchmark command used for manual verification.

## Commit & Pull Request Guidelines
Recent history favors short, scope-first subjects such as `progress/activation` and `progress/layers`. Keep commits narrow and descriptive, one logical change per commit. PRs should explain why the change is needed, list the commands you ran, call out CUDA or model assumptions, and include before/after benchmark numbers for attention, scheduler, or throughput changes.

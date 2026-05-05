# Repository Guidelines

## Project Structure & Module Organization
`mini-vllm/` is the active codebase. Core package code lives in `mini-vllm/src/myvllm/`: use `engine/` for scheduling and KV-cache flow, `layers/` for attention and layer primitives, `models/` for model definitions, and `utils/` for shared helpers. Runnable demos and benchmarks sit beside the package in `mini-vllm/main.py`, `main_llama32.py`, `benchmark_prefilling.py`, `benchmark_decoding.py`, and `benchmark_tps.py`. Tests live in `mini-vllm/tests/`. Top-level `docs/` contains study notes. `vLLM/vllm-upstream/` and `vLLM/vllm-ascend/` are reference/vendor trees; avoid incidental edits unless the task explicitly targets them.

## Build, Test, and Development Commands
Work from `mini-vllm/`.

- `uv sync`: install Python 3.11 dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py`: run the main Qwen3 demo pipeline.
- `uv run python benchmark_prefilling.py`: compare prefilling attention implementations.
- `uv run python benchmark_decoding.py`: compare decoding/PageAttention implementations.
- `uv run pytest tests/test_scheduler.py`: run the current regression suite.
- `uv run black src tests && uv run isort src tests`: format code and imports before review.

## Coding Style & Naming Conventions
Use 4-space indentation, keep modules focused, and add type hints on public APIs. Follow standard Python naming: `snake_case` for functions, files, and variables; `PascalCase` for classes; `UPPER_CASE` for constants. Keep changes aligned with existing boundaries: scheduling logic belongs in `engine/`, tensor math in `layers/` or `models/`. Prefer small, surgical edits over broad refactors.

## Testing Guidelines
Use `pytest` and place new tests under `mini-vllm/tests/` as `test_<feature>.py`. Name cases for behavior, for example `test_preempt_only_seq_when_cant_append`. No formal coverage threshold is defined, so every bug fix should ship with a regression test. Prioritize scheduler state transitions, block allocation/preemption, and benchmark-facing logic that could change outputs or token counts.

## Commit & Pull Request Guidelines
Recent history favors short, scoped subjects such as `fix`, `progress/layers`, `更新文档`, and `refactor/调整为学习版的文件结构`. Keep commit titles concise and imperative; an optional scope prefix like `fix/`, `docs/`, `refactor/`, or `progress/` matches the existing pattern. Pull requests should state the motivation, list touched paths, include exact verification commands, and attach benchmark results when performance-sensitive code changes.

## Configuration Tips
The package targets Python `>=3.11,<3.12` and assumes a CUDA-capable environment for most demos and benchmarks. Avoid hardcoding local model cache paths in reusable code; prefer config values or environment-specific overrides.

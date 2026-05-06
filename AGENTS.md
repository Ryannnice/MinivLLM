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
- `uv run black src tests`: format Python code.
- `uv run isort src tests`: sort imports.

## Coding Style & Naming Conventions

Use 4-space indentation, focused modules, and type hints on public APIs. Follow standard Python naming: `snake_case` for functions, files, and variables; `PascalCase` for classes; `UPPER_CASE` for constants. Keep scheduling logic in `engine/`, tensor math in `layers/` or `models/`, and shared helpers in `utils/`. Prefer small, surgical edits over broad refactors.

## Testing Guidelines

Use `pytest`. Place new tests under `mini-vllm/tests/` as `test_<feature>.py`, with behavior-focused names such as `test_preempt_only_seq_when_cant_append`. Every bug fix should include a regression test when practical. Prioritize scheduler state transitions, block allocation/preemption, and benchmark-facing logic that can change outputs or token counts.

## Commit & Pull Request Guidelines

Recent history uses short, scoped subjects such as `fix/修正文档`, `progress/layers`, `更新文档`, and `refactor/调整为学习版的文件结构`. Keep commit titles concise and imperative; optional prefixes like `fix/`, `docs/`, `refactor/`, or `progress/` match the existing pattern. Pull requests should state the motivation, list touched paths, include exact verification commands, and attach benchmark results for performance-sensitive changes.

## Configuration Tips

The package targets Python `>=3.11,<3.12` and assumes a CUDA-capable environment for most demos and benchmarks. Avoid hardcoding local model cache paths in reusable code; prefer configuration values or environment-specific overrides.

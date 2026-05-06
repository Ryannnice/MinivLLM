# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo contains three top-level trees serving different purposes:

- `mini-vllm/` — **the active codebase**. A learning-oriented, custom vLLM implementation targeting Qwen3 and Llama 3.2 with custom PageAttention / FlashAttention paths. All edits should land here unless a task explicitly targets the vendor trees. The package itself uses a **src layout**: importable code lives at `mini-vllm/src/myvllm/`, installed via `setup.py` + `pyproject.toml`. Demos and benchmarks (`main.py`, `benchmark_*.py`) sit at `mini-vllm/` top-level alongside `tests/`.
- `vLLM/vllm-upstream/` and `vLLM/vllm-ascend/` — **reference/vendor trees** of upstream vLLM and the Ascend backend. Treat as read-only references; do not make incidental edits.
- `docs/` — Chinese-language study notes about vLLM and Ascend internals (`vLLM.md`, `vLLM-Ascend.md`, `拆解.md`, etc.).
- `mini-vllm/HowToApproachvLLM_zh.md` — step-by-step Chinese walkthrough of how this minimal vLLM is built layer-by-layer (activations → linear → RoPE → attention → KV cache → scheduler). Read this first when picking up an unfamiliar subsystem; it doubles as the project's design rationale. `mini-vllm/README_zh.md` covers the quick-start and per-script purpose.

## Common Commands

Run from `mini-vllm/`. The project uses `uv` and pins Python `>=3.11,<3.12`.

- `uv sync` — install/sync dependencies from `pyproject.toml` + `uv.lock`.
- `uv run python main.py` — Qwen3-0.6B end-to-end inference demo.
- `uv run python main_llama32.py` — Llama-3.2-1B-Instruct demo.
- `uv run python benchmark_prefilling.py` — prefill attention comparison (PyTorch O(N²) vs Naive Triton vs FlashAttention).
- `uv run python benchmark_decoding.py` — decode-stage PageAttention comparison (Naive PyTorch vs Optimized PyTorch vs Triton kernel).
- `uv run python benchmark_tps.py` — end-to-end tokens/sec benchmark.
- `uv run pytest tests/test_scheduler.py` — current regression suite. Single test: `uv run pytest tests/test_scheduler.py::test_<name>`.
- `uv run black src tests && uv run isort src tests` — format before review.

The demos assume a CUDA GPU and may load model weights from `~/huggingface/<model>/` (see `main.py`).

## Architecture (mini-vllm)

The `myvllm` package mirrors vLLM's layered design but is intentionally compact for study. The control flow is:

```
LLMEngine.generate(prompts, sampling_params)
  └─ add_prompt → Scheduler.add_sequence (waiting queue)
  └─ loop: step()
       ├─ Scheduler.schedule()  → picks prefill or decode batch
       │     ├─ prefill: drains waiting queue while BlockManager.can_allocate
       │     └─ decode:  iterates running queue, preempts tail when can_append fails
       ├─ ModelRunner.call("run", seqs, is_prefill)  → forward + sample
       └─ Scheduler.postprocess()  → append token, check EOS / max_tokens / max_model_length, deallocate finished
```

Key boundaries to respect when editing:

- **`engine/`** owns scheduling + KV state. `scheduler.py` produces a *prefill* batch OR a *decode* batch each `step()`, never both. `block_manager.py` owns block allocation, `can_allocate`/`can_append`/`append`/`deallocate`. `sequence.py` defines `Sequence` and `SequenceStatus` (WAITING/RUNNING/FINISHED). `model_runner.py` selects the model class by directory name (`Qwen3-0.6B` / `Llama-3.2-1B-Instruct`) and runs the forward pass; in multi-GPU mode it spawns worker processes via `torch.multiprocessing.spawn` and uses NCCL on `tcp://localhost:12345`.
- **`layers/`** holds tensor primitives: `attention.py` (prefill FlashAttention + decode PageAttention dispatch), `rotary_embedding.py`, `linear.py`, `layernorm.py`, `activation.py`, `embedding_head.py`, `sampler.py`. New attention kernels belong here.
- **`models/`** wires layers into model definitions (`qwen3.py`, `llama.py`). Model selection in `ModelRunner.__init__` is a `match` on the basename of `model_name_or_path` — adding a model means adding both a class here and a case there.
- **`utils/`** has `context.py` (per-step global context shared with kernels) and `loader.py` (HF weight loading).
- `sampling_parameters.py` defines `SamplingParams` (temperature, `max_tokens`, `max_model_length`, `ignore_eos`).

### Distributed initialization order
In `LLMEngine.__init__`, `ModelRunner` is constructed before `Scheduler`. When `world_size > 1`, `ModelRunner.__init__` calls `dist.init_process_group()` which is a collective barrier — rank-0 blocks until workers join. The `Scheduler` must be created only after that rendezvous; do not reorder these.

### Prefill vs decode separation
`Scheduler.schedule()` returns `(seqs, is_prefill)`. Tensor shapes and the kernels selected in `layers/attention.py` differ between the two; any change to batch construction must keep the prefill/decode split intact.

### Config dict
`main.py` and `main_llama32.py` carry an explicit config dict (vocab/hidden/heads/kv_heads/head_dim/rope base/etc.). Comments in `main.py` flag values that diverge from older defaults to match HF (`vocab_size=151936`, `head_dim=128`, `base=1000000`, `ffn_bias=False`, `eos=151645`); preserve those when touching demo configs.

## Conventions

`AGENTS.md` is a parallel agent-instruction file with overlapping content; the rules below are the canonical short list — no need to also load `AGENTS.md` unless a task asks for it.

- Keep changes inside the existing module boundaries (scheduling in `engine/`, tensor math in `layers/` or `models/`). Prefer surgical edits to broad refactors.
- 4-space indent, type hints on public APIs, `snake_case` / `PascalCase` / `UPPER_CASE`.
- New tests go under `mini-vllm/tests/` as `test_<feature>.py`; name cases for behavior (e.g. `test_preempt_only_seq_when_cant_append`). Prioritize scheduler state transitions, block allocation/preemption, and benchmark-facing logic.
- Commit subjects are short, imperative, often with a scope prefix: `fix/`, `docs/`, `refactor/`, `progress/` (e.g. `progress/layers`, `refactor/调整为学习版的文件结构`). Mixed Chinese/English subjects are normal.
- Do not hardcode local model cache paths in reusable code; use config or env overrides.

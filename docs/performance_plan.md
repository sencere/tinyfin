# tinyfin Performance & Device Plan (Milestone 6)

This document outlines a pragmatic path to optional performance improvements and device work while keeping the current CPU-only core stable.

## Backend Abstraction (initial design)
- Introduce a thin backend vtable (`struct Backend { name, matmul, conv2d, ... }`) with a global `backend_get()/backend_set()` accessor. Python bindings already dispatch `py_matmul/py_conv2d` via the backend if provided.
- Default backend: existing CPU kernels (backend unset => CPU path).
- Optional backends (future): CUDA stub that forwards to cuBLAS/cuDNN, or a BLAS-backed CPU matmul.
- Keep the Python surface unchanged; backends are selected via an environment variable or a small C API call.
- Environment selection: `TINYFIN_BACKEND=name` picked up at init if the backend is registered; falls back to CPU if unknown.

## Optional CUDA Kernels
- Start with matmul and conv2d bindings to cuBLAS/cuDNN (or CUTLASS) behind the backend interface.
- Guard with compile-time flag (`TINYFIN_ENABLE_CUDA`) and runtime device checks.
- Reuse existing tensor device field to route ops to the backend.
- Minimal baseline implemented: a naive CUDA matmul kernel (host->device copy, strided, float32-only) guarded by `ENABLE_CUDA`/`TINYFIN_ENABLE_CUDA`. Output keeps `DEVICE_GPU` flag; backward currently runs on CPU. Conv2d/elementwise remain TODO.

## Reduced Allocations
- Add scratch allocator hooks to reuse temporary buffers in matmul/conv and some elementwise ops.
- Investigate pooling and reduction ops for in-place reuse of gradient buffers where safe.
- Profile autograd graph creation to avoid redundant small allocations.
- Implementation sketch: add a global `scratch_alloc(size_t bytes)`/`scratch_free_all()` API with a configurable arena size; route matmul/conv temporaries through it under a flag (fallback to malloc).
- Status: scratch arena implemented (env `TINYFIN_SCRATCH_BYTES`, default 1MB), matmul/conv reuse it, reset per op; unit test added to ensure reuse/malloc fallback behavior. Not thread-safe yet.

## In-place Backward Ops
- Identify safe in-place paths for elementwise ops (add/mul/sub/div) where gradient formulas do not require the original input.
- Gate with a debug flag to compare against the current out-of-place backward for correctness.

## Optional Multithreading
- Start with OpenMP or a tiny thread pool for matmul/conv2d and reductions.
- Add `TINYFIN_THREADS=N` env var to control parallelism; default to single-threaded for determinism.
- Keep a deterministic mode switch to disable threading in CI.

## Next Steps (actionable)
- [ ] Add backend interface header + CPU backend registration (no behavior change).
- [ ] Add backend registry/selector (done for matmul/conv2d; Python getter/setter available; env var switch supported).
- [ ] Add BLAS-backed matmul as an optional backend implementation.
- [~] Add scratch allocator for matmul/conv temporaries (arena scaffold present; not yet wired).
- [ ] Prototype OpenMP/thread-pool parallelism for matmul and reductions.
- [ ] Wire backend selection from Python via a simple setter/getter.
- [~] CUDA backend stub: registered as `cuda` when `TINYFIN_ENABLE_CUDA_STUB` is defined; currently falls back to CPU.
- [~] BLAS backend stub: registered as `blas` when `TINYFIN_ENABLE_BLAS_STUB` is defined; currently falls back to CPU.

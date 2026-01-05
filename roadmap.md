# tinyfin Roadmap

**Goal:**  
Bring **tinyfin** to a *TinyTorch-complete* state: a small but real ML framework with autograd, neural network layers, optimizers, and a usable Python API.

This document describes **what** remains to be built and the order of execution.

---

## Milestone 0 — Foundation (Done)

- [x] Core tensor abstraction (shape, dtype, device)
- [x] Elementwise ops + broadcasting
- [x] First-order autograd engine
- [x] Stable softmax + cross-entropy
- [x] CI with C + Python tests

---

## Milestone 1 — Core Tensor & Autograd (Mostly Done)

### Tensor System
- [x] `float32` / `float64`
- [x] Explicit device field (`DEVICE_CPU`, `DEVICE_GPU` placeholder)
- [x] Device propagation and consistency checks
- [x] In-place ops (`add_`, `sub_`, `mul_`, `div_`)

### Autograd
- [x] Broadcasting-aware backward
- [x] Gradient accumulation correctness
- [x] Multiple `backward()` calls
- [x] `retain_graph`
- [x] `no_grad` (C + Python)

---

## Milestone 2 — Neural Network Stack (Mostly Done)

### Modules & Parameters
- [x] `Parameter` abstraction
- [x] Recursive `parameters()`
- [x] `train()` / `eval()`
- [x] `zero_grad()`

### Layers
- [x] Linear
- [x] Conv2D
- [x] Flatten
- [x] Embedding
- [x] Sequential

### Activations
- [x] ReLU, Sigmoid, Tanh, LeakyReLU
- [x] Softmax, LogSoftmax

---

## Milestone 3 — Training & Optimization (In Progress)

### Losses
- [x] MSE, L1, Huber
- [x] Cross-Entropy (logits)
- [x] BCE / BCE with logits
- [x] `none`, `sum`, `mean` reductions
- [x] Weighted reductions

### Optimizers
- [x] SGD
- [x] SGD + Momentum
- [x] Adam
- [x] RMSProp
- [x] Weight decay (per-optimizer)
- [x] Per-parameter options
- [x] Optimizer state serialization (SGD/Adam/RMSProp)

### Training Utilities
- [x] DataLoader
- [x] Gradient clipping (global norm scaling in optimizers)
- [x] Dataset abstraction
- [x] Learning-rate schedulers (Step, Exponential, Warmup implemented)

---

## Milestone 4 — Stability, Safety, and Ergonomics (High Priority)

### Memory Safety (C)
- [x] Deterministic tensor destruction
- [ ] Valgrind-clean test suite
- [ ] Refcount or arena allocator
- [ ] No backward-path leaks

### Numerical Stability
- [x] Log-sum-exp
- [x] Consistent epsilon handling (clamp_min helper + log/div guards)
- [x] NaN / Inf detection
- [~] Overflow tests (exp/log/div/softmax covered; broaden coverage)
- [x] Finiteness check helper (assert_finite)

### API Ergonomics
- [ ] Chainable ops
- [ ] Function + method APIs
- [x] PyTorch-like naming
- [x] Clean Python bindings
- [~] Reshape exposed with size validation in Python API

---

## Milestone 5 — Higher-order Autograd (Planned)

- [ ] Autograd-aware backward intermediates (retain needed tensors safely; debug flag to compare with current freeing strategy)
- [~] VJP helper (prototype `tinyfin.vjp`; C API/JVP TBD)
- [~] JVP helper (finite-diff prototype `tinyfin.jvp`; forward/reverse-mode version TBD)
- [~] HVP helper (finite-diff prototype `tinyfin.hvp`; true higher-order path TBD)
- [ ] Numeric second-order gradient checks (scalar+vector cases; small shapes/dtypes)
- [ ] Minimal Python exposure (experiments module; keep core API stable)
- Notes: start with a gated prototype (`TINYFIN_ENABLE_HIGHER_ORDER`) to avoid regressions; add a tiny test harness that compares finite-diff Hessian slices vs. autodiff for small matmul/activation chains.
- Plan:
  - [ ] Gate a persistent-tape mode for higher-order tests only; keep default free-on-backward behavior.
  - [ ] Implement `vjp(f, x, v)` and `jvp(f, x, v)` helpers in C with a slim Python wrapper (Python prototypes exist: `tinyfin.vjp`, finite-diff `tinyfin.jvp/hvp`).
  - [ ] Add Hessian-vector product helper using double VJP and validate vs finite differences on small shapes (matmul + activation).
  - [ ] Document memory/perf caveats; keep feature off by default.

## Near-Term Priorities (from “What’s Missing”)
- Memory safety: Valgrind/ASAN/UBSAN runs; document ownership/refcount plan.
- API hardening: versioning/deprecation strategy; consistent error messages.
- Serialization: strict validation + forward-compat story.
- Data: pure-Python loaders for MNIST/CIFAR (prefetch/workers plan).
- Performance: BLAS baseline wired by default when available; CUDA conv/elementwise kernels; profiling/reporting.
- Coverage: transpose/concat/stack/pad (added), split (Python-level), grouped/depthwise conv, metrics/losses.

---

## Milestone 6 — Optional Performance & Device Work

- [~] Backend abstraction (design plan drafted in docs/performance_plan.md; Python matmul/conv2d dispatch through backend hooks)
- [~] Optional CUDA kernels (matmul, conv, elementwise) — naive CUDA matmul kernel available behind `ENABLE_CUDA`/`TINYFIN_ENABLE_CUDA`; conv/elementwise still fall back to CPU (stubs remain).
- [~] Reduced allocations (scratch arena with per-op reset; matmul/conv2d reuse scratch buffers; unit test added; helper to detect arena allocations to avoid double-free)
- [ ] In-place backward ops
- [~] Optional multithreading (OpenMP toggle for matmul via `TINYFIN_THREADS`; default single-thread)
- Notes: backend registry with env var selection (`TINYFIN_BACKEND`); default CPU backend registered; CUDA backend registers when enabled and runs matmul on GPU (host copy in/out); BLAS/CUDA stubs still available for fallback experimentation.

---

## Milestone 7 — Framework Hardening (Planned)

### API & Packaging
- [x] Stable module layout (`tinyfin.tensor/nn/optim/utils`)
- [ ] Versioning + deprecation policy
- [~] Clearer error messages (shape/dtype/device checks added for core ops; broaden coverage)
- [~] API reference + type hints (stub docs added; needs full coverage)

### Data & Training Ergonomics
- [x] Dataset abstraction + transforms
- [~] DataLoader workers/prefetch/sharding + deterministic seeding (seeding/shuffle done; workers/prefetch TBD)
- [~] Callbacks/hooks (logging + checkpointing scaffold; early stopping TBD)
- [x] Learning-rate schedulers (step/exp/warmup implemented)
- [x] Gradient accumulation
- [ ] Mixed precision (optional)
- [x] Example scripts (MNIST/CIFAR/transformer)

### Serialization
- [ ] Versioned checkpoints with shape/dtype validation
- [~] Optimizer/scheduler state save/load (optimizers SGD/Adam/RMSProp done; scheduler state TODO)
- [ ] Migration/forward-compatibility plan

### Testing & Stability
- [ ] Property-based tests (random shapes/dtypes/broadcast)
- [ ] Broader overflow/underflow coverage across ops
- [ ] Expanded gradient checks

### Memory Safety
- [ ] Valgrind/ASAN/UBSAN clean runs
- [ ] Ownership/lifetime documentation
- [ ] Refcount or arena allocator (evaluate)
- [ ] Thread-safety stance documented
- [ ] Serialization robustness: strict shape/dtype validation + forward-compat plan
- [ ] API/versioning: contracts + deprecation policy
- [ ] Error messages: consistent, actionable shape/dtype/device errors

### Performance & Device
- [~] BLAS-backed matmul baseline (optional `ENABLE_BLAS`; matmul via cblas_sgemm)
- [ ] Threading/vectorization + reduced allocations
- [~] Optional GPU backend + profiling hooks (CUDA matmul prototype; perf profiling script added)
- [~] BLAS backend conv2d (im2col + GEMM float32) when `TINYFIN_BACKEND=blas`

### Ecosystem & Release
- [ ] Docs/tutorials; contributor guide/formatting/linting
- [ ] Release process + semantic versioning
- [ ] Platform/Python support matrix
- [~] Examples: add runnable CPU/CUDA samples + perf profiler + synthetic MNIST/CIFAR/Transformer demos
- [~] Dataset/transform ecosystem: built-in loaders for MNIST/CIFAR (pure Python), prefetch/workers roadmap
- [ ] Type hints + API reference expansion

---

## Milestone 8 — Next Sprint (pragmatic sequence)

### API & Packaging
- [x] Lock stable module layout (`tinyfin.tensor/nn/optim/utils`) and top-level exports.
- [x] Add initial type hints and a short API reference stub (README/docs).
- [~] Improve Python-side error messages for shape/device mismatches (added type/device/shape checks for add/sub/mul/div/matmul/conv2d/embedding; expand coverage; device getters/setters exposed).

### Data & Training Ergonomics
- [x] Minimal `Dataset` + composable `Transform` interfaces.
- [x] DataLoader improvements: deterministic seeding + shuffle (workers/prefetch TBD).
- [~] Lightweight callbacks: logging + checkpointing hooks (basic callback/Trainer + Logging/Checkpoint callbacks; schedulers: StepLR implemented).

### Serialization
- [ ] Versioned checkpoint format with shape/dtype validation.
- [~] Versioned checkpoint format with shape/dtype validation (basic JSON checkpoint utilities added with magic + shape-length/dtype/checksum validation).
- [x] Save/load optimizer state (SGD/Adam/RMSProp done; schedulers persist state)

### Testing & Stability
- [~] Property-based tests for shapes/dtypes/broadcasting (initial random-shape add/mul/matmul checks added; expand harness/ops).
- [~] Broaden overflow/underflow and gradient checks to additional ops (added device/shape error tests for matmul/mul/div/add/conv2d/batchnorm/linear/squeeze; random shape add/sub/mul/div/matmul/log/exp/softmax/log_softmax checks; overflow stability checks for exp/log/softmax).
- [~] Reshape size validation/tests added; pooling ops (max/avg) exposed with shape validation.

### Memory Safety & Performance (parallel)
- [ ] Run Valgrind/ASAN on C tests; triage leaks/UB.
 - [~] BLAS-backed matmul toggle (`ENABLE_BLAS`/`TINYFIN_BACKEND=blas`); document thread-safety stance.

---

## Near-Term Checklist

- [x] Implement actual gradient clipping (compute global norm and scale grads).
- [x] Add scheduler state save/load alongside optimizer state.
- [x] Add gradient accumulation option in `Trainer`.
- [~] Expand device/shape checks to remaining ops; broaden overflow tests (initial device/shape error tests added).
- [ ] Draft versioned checkpoint format (tensor metadata + optimizer/scheduler state). (partially covered by JSON checkpoint helper)
- [ ] Prototype in-place backward for safe elementwise ops (flagged/compare vs baseline).
  - [~] Add BLAS conv2d (im2col+GEMM float32); CUDA conv/elementwise remain TODO; document deterministic/threading stance.

## Definition of “Complete”

**tinyfin is TinyTorch-complete when it has:**
- First-order autograd
- Core tensor ops + activations
- Linear + Conv layers
- Standard losses
- SGD + Adam
- Training loop utilities
- Gradient checking

To feel like a *full framework*, we additionally need the Milestone 7 items above (API polish, data ecosystem, serialization, stability testing, safety/performance, and docs/release process).

---

## Current Status (Jan 2026)

- Autograd engine stable and tested (log/div epsilon guards; exp overflow clamp).
- NN layers and optimizers implemented; momentum + weight decay supported.
- Optimizer state save/load for SGD/Adam/RMSProp; schedulers (step/exp/warmup) with state save/load.
- Training ergonomics: DataLoader with deterministic shuffle, callbacks scaffold, Trainer with gradient accumulation, checkpoint helper (tensors + opt/scheduler).
- Pooling/reshape exposed in Python with shape validation; device/shape checks added for conv2d/linear/batchnorm/broadcast ops.
- Stability tests cover log/softmax/div/exp overflow paths; random-shape tests for add/sub/mul/div/matmul/log/exp/softmax/log_softmax.
- Backend registry with env selection (`TINYFIN_BACKEND`), default CPU; optional CUDA backend with naive matmul kernel; scratch arena (per-op reset + test) used by matmul/conv2d.
- CI available; Python tests under `tests/python` cover schedulers, optimizer state, dataloader, callbacks, gradient clipping/accumulation, pooling, overflow, and validation guards.

---

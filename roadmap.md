# tinyfin Roadmap (Concise)

**Goal:** TinyTorch-like core: tensors + autograd + NN layers + optimizers + usable Python API.

**Next up (one sprint):**
- Scratch allocator follow-ups (size tuning, reuse instrumentation).
- Backend parity tests: extend coverage beyond CUDA where available.
- Backend residency: reduce host<->device copies for CUDA paths.

## Core Engine

### Milestone 0–3 (Done/Mostly Done)
- Core tensors (`float32/64`, device field), elementwise/broadcast ops, first-order autograd.
- NN stack: Linear/Conv2D/Embedding/Sequential, activations, losses, DataLoader, schedulers, optimizers (SGD/Momentum/Adam/RMSProp) with state save/load.
- Training utilities: callbacks, Trainer, gradient clipping/accumulation, no_grad/retain_graph.
- Acceptance:
  - [x] Core tensor ops + broadcasting pass basic C/Python tests.
  - [x] Autograd works on small graphs with numerical grad checks.
  - [x] Minimal training loop runs end-to-end (loss decreases).

### Milestone 4 — Stability & Ergonomics (Done)
- Memory safety: ASAN/UBSAN build targets, valgrind helper, ownership notes; refcount/arena (TBD).
- Numerical stability: overflow/log clamp tests (C + Python), finiteness checks (core done; broaden).
- API ergonomics: clearer errors for common losses, method/functional aliases (`relu`, `dot`, `item`), chainable ops/function+method APIs.
- API convenience: random constructors, init helpers, `flatten` alias, `relu` binding, loss aliases for common names.
- Acceptance:
  - [x] ASAN/UBSAN builds run a minimal test set without errors.
  - [x] Overflow/NaN guards tested in C + Python.
  - [x] Error messages for common loss misuse validated in tests.

### Milestone 5 — Higher-Order Autograd (Planned)
- Persistent tape gate, VJP/JVP/HVP helpers, finite-diff checks, scoped to experiments.
- Acceptance:
  - [x] Toggleable tape retention mode with tests.
  - [x] VJP/JVP/HVP helpers verified against finite-diff.
  - [x] Docs note limits (performance/memory) and scope.

### Milestone 5.5 — JIT & Graph Compilation (Done)
- Graph capture/export API for compute graphs (forward/backward).
- Simple graph compiler: op fusion passes, constant folding, shape-aware specialization.
- Execution cache + fallback to eager mode.
- Optional AOT export for static inference graphs.
- Acceptance:
  - [x] Forward graph capture with deterministic op ordering.
  - [x] Basic fusion/const-folding pass with tests.
  - [x] Cache key includes shapes/dtypes/device; eager fallback verified.

## Acceleration

### Milestone 6 — Performance & Device
- Backend registry (`TINYFIN_BACKEND`); CPU default. CUDA: matmul, broadcast add/mul, conv2d (backward on CUDA, host-resident tensors). BLAS: matmul/conv2d on CPU when selected. OpenGL/Vulkan stubs registered.
- Scratch allocator to reduce temps; guarded in-place add/mul. Optional OpenMP matmul.
- Next: CUDA resident buffers; extend multithreading/vectorization.
- Acceptance:
  - [x] Backend selection works via env + API with tests.
  - [x] CPU vs CUDA parity tests for matmul/add/mul/conv2d (fwd/bwd when available).
  - [x] Perf profiler runs on CPU + CUDA when built.

### Milestone 7 — Hardening (Done)
- Versioning/deprecation strategy, serialization format/validation, property-based tests, broader overflow/grad checks.
- Docs/tutorials, contributor guide, release process, platform/Python support matrix.
- Mixed precision: stubbed autocast in Python; real backend support TBD.
- Acceptance:
  - [x] Versioning/deprecation policy documented.
  - [x] Serialization format validated with fuzz/prop tests.
  - [x] Platform/Python support matrix published.

## Product Surface

### Milestone 8 — Near-Term Sequencing (Done)
- API polish (errors/type hints), DataLoader workers + prefetch, callbacks/hooks, dataset/transform ecosystem.
- Versioned checkpoints (tensor + optimizer/scheduler metadata).
- RNN layers: vanilla RNN, LSTM, and minGRU (core ops + Python API), with examples and basic tests.
- Coverage checklist: CNNs, MLPs, and RNNs are all first-class (docs + examples + tests).
- Acceptance:
  - [x] DataLoader workers/prefetch covered by tests.
  - [x] Checkpoints round-trip optimizer/scheduler metadata.
  - [x] RNN/LSTM/minGRU examples run and basic tests pass.

### Milestone 9 — Engine Expansion & Docs
- Backends: bring CUDA to parity (mixed precision, device residency); prototype real OpenGL/Vulkan kernels.
- Contracts/tests: backend selection/fallback semantics; integration tests across CPU/CUDA/OpenGL/Vulkan (extend coverage).
- Examples: backend-aware demos (MNIST/CIFAR CNN, transformer/BERT-style, text gen, perf profiler).
- Docs: backend/engine notes, mixed-precision guide, expanded API/how-to content.
- Acceptance:
  - [ ] Backend support matrix with ops x backends.
  - [ ] Integration tests run across CPU/CUDA (where available).
  - [ ] Docs/tutorials cover backend selection and mixed precision.

## Current Status
- Autograd and NN stack stable; device/shape validation on core ops.
- Backend selection works; CUDA/BLAS usable when built; GL/Vulkan stubs in place.
- CUDA conv2d backward supported (host-resident tensors); BLAS dispatch works for CPU float32.
- Tests cover schedulers, optimizer state, dataloader, callbacks, overflow/validation, backend selection, backend parity (CUDA), mixed-precision stub.
- Perf profiler script validated on CPU and CUDA (when built).
- Graph export now includes shape/dtype metadata; Python IR + cache-key helpers available.
- Basic const-folding and cache helpers are available for experimentation.
- Checkpoint serialization fuzz coverage added for roundtrip validation.
- Examples: MNIST/CIFAR/transformer, backend-aware CNN; perf profiler and CUDA demo available.
- Ergonomics: `Tensor.from_numpy`, `Trainer.fit`, `MLP/Flatten/MaxPool2d`, `CrossEntropyLoss`, and dataset helpers (`TensorDataset/DataLoader.from_numpy`) to reduce example boilerplate.

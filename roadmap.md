# tinyfin Roadmap (Concise)

**Goal:** TinyTorch-like core: tensors + autograd + NN layers + optimizers + usable Python API.

## Milestone 0–3 (Done/Mostly Done)
- Core tensors (`float32/64`, device field), elementwise/broadcast ops, first-order autograd.
- NN stack: Linear/Conv2D/Embedding/Sequential, activations, losses, DataLoader, schedulers, optimizers (SGD/Momentum/Adam/RMSProp) with state save/load.
- Training utilities: callbacks, Trainer, gradient clipping/accumulation, no_grad/retain_graph.

## Milestone 4 — Stability & Ergonomics (In progress)
- Memory safety: Valgrind/ASAN, ownership docs, refcount/arena (TBD).
- Numerical stability: overflow tests, finiteness checks (core done; broaden).
- API ergonomics: clearer errors, chainable ops/function+method APIs.

## Milestone 5 — Higher-Order Autograd (Planned)
- Persistent tape gate, VJP/JVP/HVP helpers, finite-diff checks, scoped to experiments.

## Milestone 6 — Performance & Device
- Backend registry (`TINYFIN_BACKEND`); CPU default. CUDA: matmul, broadcast add/mul, conv2d (inference, CPU backward). BLAS: matmul/conv2d. OpenGL/Vulkan stubs registered.
- Scratch allocator to reduce temps; guarded in-place add/mul. Optional OpenMP matmul.
- Next: CUDA conv2d backward + resident buffers; extend multithreading/vectorization.

## Milestone 7 — Hardening
- Versioning/deprecation strategy, serialization format/validation, property-based tests, broader overflow/grad checks.
- Docs/tutorials, contributor guide, release process, platform/Python support matrix.
- Mixed precision: stubbed autocast in Python; real backend support TBD.

## Milestone 8 — Near-Term Sequencing
- API polish (errors/type hints), DataLoader workers/prefetch plan, callbacks/hooks, dataset/transform ecosystem.
- Versioned checkpoints (tensor + optimizer/scheduler metadata).

## Milestone 9 — Engine Expansion & Docs
- Backends: bring CUDA to parity (conv2d backward, mixed precision, device residency); prototype real OpenGL/Vulkan kernels.
- Contracts/tests: backend selection/fallback semantics; integration tests across CPU/CUDA/OpenGL/Vulkan.
- Examples: backend-aware demos (MNIST/CIFAR CNN, transformer/BERT-style, text gen, perf profiler).
- Docs: backend/engine notes, mixed-precision guide, expanded API/how-to content.

## Current Status
- Autograd and NN stack stable; device/shape validation on core ops.
- Backend selection works; CUDA/BLAS usable when built; GL/Vulkan stubs in place.
- Tests cover schedulers, optimizer state, dataloader, callbacks, overflow/validation, backend selection, mixed-precision stub.
- Examples: MNIST/CIFAR/transformer, backend-aware CNN; perf profiler and CUDA demo available.

# tinyfin Architecture

This document describes **how tinyfin is structured**, the core design principles, and internal subsystems.

---

## Design Principles

- **Minimal but real**: no toy shortcuts that break autograd correctness
- **Explicit over implicit**: shapes, devices, and graph structure are visible
- **C-first core**: correctness and performance come before Python ergonomics
- **Python as orchestration**: Python is a thin, friendly layer over C

---

## High-level Architecture
Python API (tinyfin, tinyfin.nn, tinyfin.optim)
|
v
C Python bindings (py_api.c)
|
v
Core C engine
├── Tensor + memory
├── Autograd graph
├── Ops (forward + backward)
└── Optional backend dispatch


---

## Tensor System

### Tensor Structure
Each tensor contains:
- Shape + strides
- Dtype (`float32`, `float64`)
- Device (`DEVICE_CPU`, `DEVICE_GPU`)
- Data pointer
- Gradient pointer (optional)
- Autograd metadata (`grad_fn`, parents)

### Device Semantics
- Device is explicit and carried by tensors
- Mixed-device ops are rejected
- `DEVICE_GPU` is currently a logical placeholder
- Future backends will provide real device kernels

---

## Autograd Engine

### Graph Construction
- Ops create nodes when `requires_grad=True`
- Graph is a DAG of tensors + backward functions
- Broadcasting metadata is stored for backward reduction

### Backward Pass
1. Topological sort
2. Backward functions executed in reverse order
3. Gradients accumulated
4. Intermediate gradients zeroed (for repeated backward)
5. Graph optionally freed

### no_grad
- Implemented as a C-level push/pop flag
- Python context manager forwards to C

---

## Ops Implementation

- Each op has:
  - Forward implementation
  - Backward implementation
- Backward logic:
  - Handles broadcasting reductions
  - Avoids in-place mutation
  - Uses autograd-aware tensor creation where needed

---

## Neural Network Modules

### Module System (Python)
- `Module` tracks submodules and parameters
- `Parameter` wraps tensors with `requires_grad=True`
- Recursive traversal via `parameters()`
- `train()` / `eval()` propagate flags

### Layers
- Most layers are Python `Module` wrappers
- Core math ops are implemented in C
- Parameters are registered explicitly

---

## Loss Functions

- Implemented in C for correctness and stability
- Support multiple reduction modes
- Python wrappers enforce consistent APIs

---

## Optimizers

- Implemented in C
- Python wrappers manage parameter lists
- Optimizer state stored per-parameter

---

## Serialization

- Model weights saved via state dictionaries
- Optimizer state serialization planned
- Shape and dtype validation planned

---

## Memory Management

- Deterministic tensor destruction
- Explicit ownership rules
- No garbage collection in C
- Planned: refcount or arena allocator
- Planned: full Valgrind coverage

---

## Testing Strategy

### C Tests
- Unit tests for ops
- Autograd correctness tests
- Numeric gradient checking

### Python Tests
- Smoke tests for public API
- End-to-end training sanity checks

### CI
- Builds C library
- Runs C test suite
- Runs Python tests

---

## Backend & GPU (Future)

- Backend registry (CPU default) with env selector `TINYFIN_BACKEND`; CPU always registered.
- Backend hooks for `matmul/conv2d`; Python dispatch calls backend when tensors are on `DEVICE_GPU`.
- Optional CUDA backend (build with `ENABLE_CUDA=1`) implements a naive matmul kernel; conv/elementwise currently fall back to CPU. BLAS/CUDA stubs exist for experimentation.
- Optional BLAS backend (build with `ENABLE_BLAS=1`) uses `cblas_sgemm` for matmul when tensors are on CPU.
- Scratch arena to reduce allocations (env `TINYFIN_SCRATCH_BYTES`, per-op reset) used in matmul/conv.
- OpenMP optional for matmul (`TINYFIN_THREADS`), default single-thread.
- Profiling helper script (`examples/perf_profile.py`) for quick throughput/GFLOP/s checks on CPU/CUDA.
- Higher-order helpers gated by `TINYFIN_ENABLE_HIGHER_ORDER` (Python `vjp/jvp/hvp`, finite-diff where needed); default graph retention configurable via `TINYFIN_RETAIN_GRAPH`/`TINYFIN_PERSISTENT_TAPE`.
- Autograd graph export: `export_graph(tensor, path)` emits DOT for visualization; backend/device info included.

---

## Non-goals

- Full PyTorch API compatibility
- Dynamic graph compilation
- Distributed training
- Automatic mixed precision

---

## Summary

tinyfin is designed as:
- A **correct**, minimal autograd engine
- A **clean reference implementation** for ML systems
- A **hackable research playground**, not a production framework

---

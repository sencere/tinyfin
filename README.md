![tinyfin logo](images/tinyfin.png)

tinyfin is a small, educational deep learning library in C with a minimal autograd engine, a thin Python ctypes binding, and optional BLAS/CUDA backends.

This repository was previously known as `torch_slim_c` / `ctorch`. The code and API remain the same; the project name is now `tinyfin`.

Quick start

Build the native library and run tests:

```bash
# build tests and shared library
make -j2 libtinyfin.so

# run C unit tests (binaries in project root)
./tests/test_div_exp

# run Python smoke tests
python3 tests/python/test_div_exp_py.py
```

Development notes

- The C sources are in `src/` and headers in `include/tinyfin/` (legacy `include/ctorch/` kept for compatibility).
- Python package layout: core bindings in `python/tinyfin.py`, public namespace modules `tinyfin.tensor`, `tinyfin.nn`, `tinyfin.optim`, `tinyfin.utils`, `tinyfin.data` (importable via `import tinyfin`).
- Training utilities: lightweight `DataLoader`, callbacks (`LoggingCallback`, `CheckpointCallback`), `Trainer`, and schedulers (`StepLR`, `ExponentialLR`, `LinearWarmupLR`); optimizers (`SGDOpt`, `AdamOpt`, `RMSPropOpt`) expose `set_lr/get_lr` and state save/load helpers. Checkpoints can be persisted via `tinyfin.utils.save_checkpoint/load_checkpoint`. Tensor helpers include `reshape`, `squeeze/unsqueeze`, and pooling (`maxpool2d/avgpool2d`) with shape validation.

Examples

```python
from tinyfin import Tensor

x = Tensor.new([1, 3, 4, 4])
pooled = x.maxpool2d(2)          # -> shape [1, 3, 2, 2]
flat = pooled.reshape([1, 3, -1])
sq = flat.squeeze()              # removes dims of size 1
```
- Try the torch-free examples in `examples/`: synthetic MNIST MLP (`mnist_mlp.py`), CIFAR-like CNN (`cifar_cnn.py`), tiny transformer-like FFN (`transformer_tiny.py`), CUDA matmul demo, perf profiler, and autograd graph export.
- Built-in pure-Python helpers for MNIST/CIFAR live in `tinyfin.data` (`load_mnist`, `load_cifar10`); examples will download data if present, otherwise fall back to synthetic.
- The shared library built by the Makefile is `libtinyfin.so`.
- Numerical stability helpers include stable softmax/log-softmax, log/div epsilon guards, exp overflow clamp, NaN/Inf detection (`Tensor.has_nan_or_inf`) and a convenience `assert_finite` to catch divergence in training loops.
- See `docs/api.md` for a short Python API stub.
- Backend guide: `docs/backends.md` explains backend selection/fallback (CPU default; CUDA/BLAS optional; OpenGL/Vulkan stubs) and how to toggle via `TINYFIN_BACKEND`.
- Examples roadmap: `docs/examples.md` tracks planned/runnable demos (MNIST/CIFAR, transformer, text generation, perf profiler) across CPU/GPU backends.
- Mixed precision: stubbed `tinyfin.autocast` / `set_mixed_precision` live in Python for future fp16/bfloat16 support (see `docs/mixed_precision.md`).

Status / what's missing for a “full” framework:
- Memory safety hardening (Valgrind/ASAN), versioned/validated serialization, broader op/layer coverage (transpose/concat/stack, depthwise/group conv, padding), better performance kernels (CUDA conv/elementwise), dataset/prefetch workers, API/versioning/deprecation policy, richer docs/reference.

Contributing

Open issues or PRs for new ops, autograd fixes, or performance work. See `roadmap.md` for planned milestones.

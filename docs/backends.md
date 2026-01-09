# tinyfin backend guide

The backend registry (`backend.h`) lets tinyfin dispatch heavy ops (matmul/conv2d/elementwise) to alternative engines while retaining a CPU fallback.

## Available backends
- `cpu` (default): always registered; runs all ops on host.
- `cuda`: optional; currently provides matmul, broadcast-aware add/mul, and conv2d with backward handled on CUDA (inputs/outputs still live in host memory). Enable via `ENABLE_CUDA`/`TINYFIN_ENABLE_CUDA`.
- `blas`: optional; matmul/conv2d via BLAS (im2col+GEMM) on CPU for float32 tensors when selected. Enable via `ENABLE_BLAS`/`TINYFIN_BACKEND=blas`.
- `opengl` / `vulkan`: stub backends registered by default for experimentation; currently fall back to CPU and log a warning. Use them as placeholders when prototyping new GPU paths.

Select a backend via env `TINYFIN_BACKEND` or `tinyfin.backend_set("name")` in Python. If an op is unsupported, it falls back to the CPU implementation.

CUDA build/run quickstart:
- Build with CUDA support (requires `nvcc` on PATH): `make -j2 libtinyfin.so ENABLE_CUDA=1`
- Run with CUDA backend: `PYTHONPATH=python TINYFIN_BACKEND=cuda python3 examples/perf_profile.py cuda 512 512 512 20`
- Confirm local bindings: `PYTHONPATH=python python3 -c "import tinyfin; print(tinyfin.__file__)"`
- Note: CUDA elementwise add/mul broadcast path assumes contiguous inputs and will fall back to CPU if tensors are non-contiguous (e.g., views from transpose/permute).
- Parity checks: `tests/python/test_backend_parity.py` (forward) and `tests/python/test_backend_parity_backward.py` (backward).

Python helpers:
```python
from tinyfin import backend_set, backend_name

ok = backend_set("cuda")
print("set ok:", ok, "current:", backend_name())
```

Device mapping (current):
- CPU: device 0
- GPU: device 1 (placeholder for CUDA/OpenGL/Vulkan)

## Contracts and testing
- All backends must keep shapes/dtypes/devices consistent with CPU semantics.
- Unsupported ops must return NULL so the CPU path can run.
- Integration tests should assert identical numerical results for supported ops across backends when available (stub selection/retention is covered in `tests/python/test_backend_name.py` and `tests/python/test_backend_integration.py`).

## Next steps
- Make CUDA conv2d fully autograd-ready and keep tensors resident on device.
- Prototype real OpenGL/Vulkan backends using the same registry entry points.
- Add a mixed-precision gate (fp16/bfloat16) per backend with clear docs on safety/perf (stub in Python via `tinyfin.autocast`).

# Release checklist

Use this checklist before tagging a tinyfin release.

## Local verification

```bash
make clean
make -j2
./tests/test_div_exp
./tests/test_mlp
PYTHONPATH=python python3 tests/python/test_div_exp_py.py
PYTHONPATH=python python3 tests/python/test_linear.py
PYTHONPATH=python python3 examples/blob_mlp.py
PYTHONPATH=python python3 examples/transformer_block_tiny.py
```

If `pytest` is installed, run the Python suite:

```bash
PYTHONPATH=python python3 -m pytest -q tests/python
```

## Wheel artifact

Build and smoke-test a platform wheel:

```bash
python3 -m pip wheel . --no-build-isolation --no-deps -w dist
python3 -m venv /tmp/tinyfin-wheel-smoke
/tmp/tinyfin-wheel-smoke/bin/python -m pip install dist/tinyfin-*.whl numpy
/tmp/tinyfin-wheel-smoke/bin/python -c "import tinyfin; print(tinyfin.__version__); print(tinyfin.Tensor.rand([2, 3]).shape())"
```

## Optional backend checks

```bash
make clean
make -j2 ENABLE_BLAS=1
TINYFIN_BACKEND=blas PYTHONPATH=python python3 examples/perf_profile.py blas 128 128 128 5

make clean
make -j2 ENABLE_CUDA=1
TINYFIN_BACKEND=cuda PYTHONPATH=python python3 examples/cuda_matmul.py
```

Notes:
- `ENABLE_BLAS=1` requires OpenBLAS development headers and library (`cblas.h` and `libopenblas`).
- `ENABLE_CUDA=1` uses `CUDA_HOME` for headers and libraries. Override it if CUDA is not installed at `/usr/local/cuda`.

## Docs

- Confirm `README.md` quick start commands work from a clean checkout.
- Confirm every example listed in `examples/README.md` either runs on CPU or documents its backend requirement.
- Update `docs/versioning.md` and `roadmap.md` if the public API or release scope changed.

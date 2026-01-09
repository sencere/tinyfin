import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, backend_set, backend_name
import numpy as np


def _to_tensor(arr, requires_grad=False, device=None):
    t = Tensor.from_numpy(arr, requires_grad=requires_grad)
    if device is not None:
        t.set_device(device)
    return t


@pytest.mark.parametrize("op", ["add", "mul"])
def test_cuda_parity_elementwise(op):
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(0)
        a_np = rng.standard_normal(64, dtype=np.float32)
        b_np = rng.standard_normal(64, dtype=np.float32)

        a = _to_tensor(a_np, device=1)
        b = _to_tensor(b_np, device=1)
        out = a + b if op == "add" else a * b

        a_cpu = _to_tensor(a_np)
        b_cpu = _to_tensor(b_np)
        out_cpu = a_cpu + b_cpu if op == "add" else a_cpu * b_cpu

        assert backend_name() == "cuda"
        np.testing.assert_allclose(out.to_numpy(), out_cpu.to_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        backend_set("cpu")


@pytest.mark.parametrize("op", ["add", "mul"])
def test_cuda_parity_broadcast_elementwise(op):
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        b_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        a = _to_tensor(a_np, device=1)
        b = _to_tensor(b_np, device=1)
        out = a + b if op == "add" else a * b

        a_cpu = _to_tensor(a_np)
        b_cpu = _to_tensor(b_np)
        out_cpu = a_cpu + b_cpu if op == "add" else a_cpu * b_cpu

        assert backend_name() == "cuda"
        np.testing.assert_allclose(out.to_numpy(), out_cpu.to_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        backend_set("cpu")


def test_cuda_parity_matmul():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(1)
        a_np = rng.standard_normal((8, 6), dtype=np.float32)
        b_np = rng.standard_normal((6, 4), dtype=np.float32)

        a = _to_tensor(a_np, device=1)
        b = _to_tensor(b_np, device=1)
        out = a.matmul(b)

        a_cpu = _to_tensor(a_np)
        b_cpu = _to_tensor(b_np)
        out_cpu = a_cpu.matmul(b_cpu)

        assert backend_name() == "cuda"
        np.testing.assert_allclose(out.to_numpy(), out_cpu.to_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


def test_cuda_parity_conv2d():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((1, 1, 4, 4), dtype=np.float32)
        w_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)

        x = _to_tensor(x_np, device=1)
        w = _to_tensor(w_np, device=1)
        out = x.conv2d(w)

        x_cpu = _to_tensor(x_np)
        w_cpu = _to_tensor(w_np)
        out_cpu = x_cpu.conv2d(w_cpu)

        assert backend_name() == "cuda"
        np.testing.assert_allclose(out.to_numpy(), out_cpu.to_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


def test_cuda_parity_conv2d_with_bias():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((1, 2, 5, 5), dtype=np.float32)
        w_np = rng.standard_normal((3, 2, 3, 3), dtype=np.float32)
        b_np = rng.standard_normal((3,), dtype=np.float32)

        x = _to_tensor(x_np, device=1)
        w = _to_tensor(w_np, device=1)
        b = _to_tensor(b_np, device=1)
        out = x.conv2d(w, bias=b)

        x_cpu = _to_tensor(x_np)
        w_cpu = _to_tensor(w_np)
        b_cpu = _to_tensor(b_np)
        out_cpu = x_cpu.conv2d(w_cpu, bias=b_cpu)

        assert backend_name() == "cuda"
        np.testing.assert_allclose(out.to_numpy(), out_cpu.to_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


if __name__ == "__main__":
    pytest.main([__file__])

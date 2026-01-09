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
def test_cuda_parity_elementwise_backward(op):
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(3)
        a_np = rng.standard_normal(64, dtype=np.float32)
        b_np = rng.standard_normal(64, dtype=np.float32)

        a = _to_tensor(a_np, requires_grad=True, device=1)
        b = _to_tensor(b_np, requires_grad=True, device=1)
        out = a + b if op == "add" else a * b
        out.sum().backward()

        a_cpu = _to_tensor(a_np, requires_grad=True)
        b_cpu = _to_tensor(b_np, requires_grad=True)
        out_cpu = a_cpu + b_cpu if op == "add" else a_cpu * b_cpu
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(a.grad_numpy(), a_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(b.grad_numpy(), b_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        backend_set("cpu")


@pytest.mark.parametrize("op", ["add", "mul"])
def test_cuda_parity_broadcast_elementwise_backward(op):
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        b_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        a = _to_tensor(a_np, requires_grad=True, device=1)
        b = _to_tensor(b_np, requires_grad=True, device=1)
        out = a + b if op == "add" else a * b
        out.sum().backward()

        a_cpu = _to_tensor(a_np, requires_grad=True)
        b_cpu = _to_tensor(b_np, requires_grad=True)
        out_cpu = a_cpu + b_cpu if op == "add" else a_cpu * b_cpu
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(a.grad_numpy(), a_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(b.grad_numpy(), b_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        backend_set("cpu")


def test_cuda_parity_matmul_backward():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(4)
        a_np = rng.standard_normal((6, 5), dtype=np.float32)
        b_np = rng.standard_normal((5, 4), dtype=np.float32)

        a = _to_tensor(a_np, requires_grad=True, device=1)
        b = _to_tensor(b_np, requires_grad=True, device=1)
        out = a.matmul(b)
        out.sum().backward()

        a_cpu = _to_tensor(a_np, requires_grad=True)
        b_cpu = _to_tensor(b_np, requires_grad=True)
        out_cpu = a_cpu.matmul(b_cpu)
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(a.grad_numpy(), a_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.grad_numpy(), b_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


def test_cuda_parity_conv2d_backward():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((1, 1, 4, 4), dtype=np.float32)
        w_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)

        x = _to_tensor(x_np, requires_grad=True, device=1)
        w = _to_tensor(w_np, requires_grad=True, device=1)
        out = x.conv2d(w)
        out.sum().backward()

        x_cpu = _to_tensor(x_np, requires_grad=True)
        w_cpu = _to_tensor(w_np, requires_grad=True)
        out_cpu = x_cpu.conv2d(w_cpu)
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(x.grad_numpy(), x_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(w.grad_numpy(), w_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


def test_cuda_parity_conv2d_backward_larger():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal((1, 2, 5, 5), dtype=np.float32)
        w_np = rng.standard_normal((3, 2, 3, 3), dtype=np.float32)

        x = _to_tensor(x_np, requires_grad=True, device=1)
        w = _to_tensor(w_np, requires_grad=True, device=1)
        out = x.conv2d(w)
        out.sum().backward()

        x_cpu = _to_tensor(x_np, requires_grad=True)
        w_cpu = _to_tensor(w_np, requires_grad=True)
        out_cpu = x_cpu.conv2d(w_cpu)
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(x.grad_numpy(), x_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(w.grad_numpy(), w_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


def test_cuda_parity_conv2d_backward_with_bias():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((1, 2, 5, 5), dtype=np.float32)
        w_np = rng.standard_normal((3, 2, 3, 3), dtype=np.float32)
        b_np = rng.standard_normal((3,), dtype=np.float32)

        x = _to_tensor(x_np, requires_grad=True, device=1)
        w = _to_tensor(w_np, requires_grad=True, device=1)
        b = _to_tensor(b_np, requires_grad=True, device=1)
        out = x.conv2d(w, bias=b)
        out.sum().backward()

        x_cpu = _to_tensor(x_np, requires_grad=True)
        w_cpu = _to_tensor(w_np, requires_grad=True)
        b_cpu = _to_tensor(b_np, requires_grad=True)
        out_cpu = x_cpu.conv2d(w_cpu, bias=b_cpu)
        out_cpu.sum().backward()

        assert backend_name() == "cuda"
        np.testing.assert_allclose(x.grad_numpy(), x_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(w.grad_numpy(), w_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.grad_numpy(), b_cpu.grad_numpy(), rtol=1e-4, atol=1e-5)
    finally:
        backend_set("cpu")


if __name__ == "__main__":
    pytest.main([__file__])

import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, backend_set
import numpy as np


def test_cuda_conv2d_backward_matches_cpu():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        x_data = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        w_data = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)

        x = Tensor.from_numpy(x_data, requires_grad=True)
        w = Tensor.from_numpy(w_data, requires_grad=True)
        x.set_device(1)
        w.set_device(1)
        y = x.conv2d(w)
        y.sum().backward()

        xg_cuda = x.grad_numpy()
        wg_cuda = w.grad_numpy()

        x_cpu = Tensor.from_numpy(x_data, requires_grad=True)
        w_cpu = Tensor.from_numpy(w_data, requires_grad=True)
        y_cpu = x_cpu.conv2d(w_cpu)
        y_cpu.sum().backward()

        np.testing.assert_allclose(xg_cuda, x_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(wg_cuda, w_cpu.grad_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        backend_set("cpu")


if __name__ == "__main__":
    pytest.main([__file__])

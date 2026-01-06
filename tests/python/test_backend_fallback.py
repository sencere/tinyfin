import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, backend_set, backend_name
import numpy as np


@pytest.mark.parametrize("op", ["add", "mul"])
def test_backend_cuda_fallback_for_basic_ops(op):
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        a = Tensor.new([2], requires_grad=False)
        b = Tensor.new([2], requires_grad=False)
        a.numpy_view()[:] = [1.0, 2.0]
        b.numpy_view()[:] = [3.0, 4.0]
        a.set_device(1)  # mark as GPU
        b.set_device(1)
        out = a + b if op == "add" else a * b
        # Result should be correct even if backend falls back to CPU path
        expected = [4.0, 6.0] if op == "add" else [3.0, 8.0]
        assert out.shape() == [2]
        assert backend_name() == "cuda"
        for got, exp in zip(out.to_numpy(), expected):
            assert abs(float(got) - exp) < 1e-6
    finally:
        backend_set("cpu")

def test_backend_cuda_conv2d_fallback():
    ok = backend_set("cuda")
    if not ok:
        pytest.skip("cuda backend not registered (stub missing)")
    try:
        x = Tensor.new([1, 1, 3, 3], requires_grad=False)
        w = Tensor.new([1, 1, 2, 2], requires_grad=False)
        x.numpy_view()[:] = np.arange(9, dtype=np.float32).reshape(1,1,3,3)
        w.numpy_view()[:] = np.array([[[[1, 0],[0, 1]]]], dtype=np.float32)
        x.set_device(1); w.set_device(1)
        y = x.conv2d(w)
        assert y.shape() == [1, 1, 2, 2]
        np.testing.assert_allclose(y.to_numpy(), np.array([[[[4, 6],[10, 12]]]], dtype=np.float32))
    finally:
        backend_set("cpu")

def test_inplace_add_guarded_by_autograd():
    a = Tensor.new([2], requires_grad=True)
    b = Tensor.new([2], requires_grad=False)
    with pytest.raises(Exception):
        a.add_(b)  # should refuse in-place when autograd enabled


if __name__ == "__main__":
    pytest.main([__file__])

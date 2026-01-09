import os
import sys

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, backend_name, backend_set


def _conv2d_numpy(x, w):
    n, c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w
    h_out = h - kh + 1
    w_out = w_in - kw + 1
    out = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)
    for n_i in range(n):
        for oc in range(c_out):
            for ic in range(c_in):
                for ho in range(h_out):
                    for wo in range(w_out):
                        patch = x[n_i, ic, ho:ho + kh, wo:wo + kw]
                        out[n_i, oc, ho, wo] += np.sum(patch * w[oc, ic])
    return out


def _run_backend_ops(name):
    ok = backend_set(name)
    if not ok:
        return False
    assert backend_name() == name
    device_id = 1 if backend_name() == "cuda" else 0

    a_np = np.array([1.0, 2.0], dtype=np.float32)
    b_np = np.array([3.0, 4.0], dtype=np.float32)
    a = Tensor.from_numpy(a_np, device=device_id)
    b = Tensor.from_numpy(b_np, device=device_id)
    np.testing.assert_allclose((a + b).to_numpy(), a_np + b_np)
    np.testing.assert_allclose((a * b).to_numpy(), a_np * b_np)

    m_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    m = Tensor.from_numpy(m_np, device=device_id)
    n = Tensor.from_numpy(n_np, device=device_id)
    np.testing.assert_allclose(m.matmul(n).to_numpy(), m_np @ n_np)

    x_np = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    w_np = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
    x = Tensor.from_numpy(x_np, device=device_id)
    w = Tensor.from_numpy(w_np, device=device_id)
    np.testing.assert_allclose(x.conv2d(w).to_numpy(), _conv2d_numpy(x_np, w_np))

    return True


def test_backend_support_matrix_ops():
    for name in ("cpu", "blas", "cuda", "opengl", "vulkan"):
        if _run_backend_ops(name):
            backend_set("cpu")

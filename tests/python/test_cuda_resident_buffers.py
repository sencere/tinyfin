import os
import sys

import numpy as np
import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, backend_set


def test_cuda_resident_buffers_env():
    if not backend_set("cuda"):
        pytest.skip("cuda backend not available")
    os.environ["TINYFIN_CUDA_RESIDENT"] = "1"

    a_np = np.array([1.0, 2.0], dtype=np.float32)
    b_np = np.array([3.0, 4.0], dtype=np.float32)
    a = Tensor.from_numpy(a_np, device=1)
    b = Tensor.from_numpy(b_np, device=1)
    np.testing.assert_allclose((a + b).to_numpy(), a_np + b_np)

    m_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    m = Tensor.from_numpy(m_np, device=1)
    n = Tensor.from_numpy(n_np, device=1)
    np.testing.assert_allclose(m.matmul(n).to_numpy(), m_np @ n_np)

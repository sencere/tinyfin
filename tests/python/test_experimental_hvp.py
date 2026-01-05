import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

import numpy as np
import os
os.environ["TINYFIN_ENABLE_HIGHER_ORDER"] = "1"

from tinyfin import Tensor, hvp


def f_square(x: Tensor) -> Tensor:
    return x * x


def test_hvp_square_fd():
    x = Tensor.new([3], requires_grad=False)
    x.numpy_view()[:] = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    v = Tensor.new([3], requires_grad=False)
    v.numpy_view()[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    y, h = hvp(f_square, x, v, eps=1e-3)
    np.testing.assert_allclose(y.to_numpy(), np.array([1.0, 4.0, 0.25], dtype=np.float32))
    # Hessian of sum(x^2) is 2I, so H*v = 2*v
    np.testing.assert_allclose(h, np.array([2.0, 2.0, 2.0], dtype=np.float32), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_hvp_square_fd()
    print("[test_experimental_hvp.py] PASS")

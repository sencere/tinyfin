import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

import numpy as np
import os
os.environ["TINYFIN_ENABLE_HIGHER_ORDER"] = "1"

from tinyfin import Tensor, vjp


def f_square(x: Tensor) -> Tensor:
    return x * x


def test_vjp_square():
    x = Tensor.new([3], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    v = Tensor.new([3], requires_grad=False)
    v.numpy_view()[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    y, grad = vjp(f_square, x, v)
    np.testing.assert_allclose(y.to_numpy(), np.array([1.0, 4.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(grad, np.array([2.0, 4.0, -6.0], dtype=np.float32), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_vjp_square()
    print("[test_experimental_vjp.py] PASS")

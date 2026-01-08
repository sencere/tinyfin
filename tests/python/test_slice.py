import sys
import os
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def test_slice_values_and_grad():
    t = Tensor.new([3, 4], requires_grad=True)
    t.numpy_view()[:] = np.arange(12, dtype=np.float32).reshape(3, 4)
    s = t.slice(0, 1, 3)
    assert s.shape() == [2, 4]
    assert np.allclose(s.to_numpy(), np.arange(4, 12, dtype=np.float32).reshape(2, 4))

    out = s.sum()
    out.backward()
    g = t.grad_numpy()
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1:3, :] = 1.0
    assert np.allclose(g, expected)


if __name__ == '__main__':
    test_slice_values_and_grad()
    print('[test_slice.py] PASS')

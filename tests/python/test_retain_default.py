import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

import numpy as np
from tinyfin import Tensor, set_retain_graph_default, get_retain_graph_default


def test_retain_default_toggle():
    # default is retain=True (from C ctor)
    assert get_retain_graph_default() is True
    set_retain_graph_default(False)
    assert get_retain_graph_default() is False
    set_retain_graph_default(True)
    assert get_retain_graph_default() is True

    # simple backward should work under both settings
    for retain in (True, False):
        set_retain_graph_default(retain)
        x = Tensor.new([2], requires_grad=True)
        x.numpy_view()[:] = np.array([1.0, 2.0], dtype=np.float32)
        y = x * x  # elementwise square
        s = y.sum()
        s.backward()
        np.testing.assert_allclose(x.grad_numpy(), np.array([2.0, 4.0], dtype=np.float32))


if __name__ == "__main__":
    test_retain_default_toggle()
    print("[test_retain_default.py] PASS")

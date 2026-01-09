import os
import sys
import tempfile

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, export_graph


def _build_graph():
    x = Tensor.new([3], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    y1 = (x * x).sum()
    y2 = (x + x).sum()
    return y1 + y2


def test_export_graph_deterministic():
    y = _build_graph()
    with tempfile.NamedTemporaryFile(suffix=".dot") as f1, tempfile.NamedTemporaryFile(suffix=".dot") as f2:
        export_graph(y, f1.name)
        export_graph(y, f2.name)
        s1 = f1.read()
        s2 = f2.read()
        assert s1 == s2

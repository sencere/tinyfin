import os
import sys
import tempfile

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, export_graph


def test_export_graph_smoke():
    x = Tensor.new([2], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, -2.0], dtype=np.float32)
    y = (x * x).sum()
    with tempfile.NamedTemporaryFile(suffix=".dot") as tmp:
        export_graph(y, tmp.name)
        data = tmp.read()
        assert b"digraph autograd" in data

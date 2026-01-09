import os
import sys

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, export_graph_ir


def test_export_graph_ir_contents():
    x = Tensor.new([2], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, -2.0], dtype=np.float32)
    y = (x * x + x).sum()
    ir = export_graph_ir(y)
    assert len(ir["nodes"]) > 0
    assert len(ir["edges"]) > 0
    for node in ir["nodes"]:
        assert "device" in node
        assert "dtype" in node
        assert "shape" in node

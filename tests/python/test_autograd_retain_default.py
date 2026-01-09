import os
import sys
import tempfile

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, export_graph, set_retain_graph_default


def _make_graph():
    x = Tensor.new([3], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = (x * x).sum()
    return x, y


def test_retain_default_false_frees_graph():
    set_retain_graph_default(False)
    x, y = _make_graph()
    y.backward()

    with tempfile.NamedTemporaryFile(suffix=".dot") as tmp:
        failed = False
        try:
            export_graph(y, tmp.name)
        except RuntimeError:
            failed = True
        assert failed, "expected graph export to fail after backward when retain default is False"


def test_retain_default_true_keeps_graph():
    set_retain_graph_default(True)
    x, y = _make_graph()
    y.backward()

    with tempfile.NamedTemporaryFile(suffix=".dot") as tmp:
        export_graph(y, tmp.name)
        data = tmp.read()
        assert b"digraph autograd" in data

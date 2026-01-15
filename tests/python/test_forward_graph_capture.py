import os
import sys

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, capture_graph, graph_signature, compile_graph, graph_exec_cache_run


def _make_inputs():
    x = Tensor.new([2, 3], requires_grad=False)
    w = Tensor.new([3, 4], requires_grad=False)
    b = Tensor.new([4], requires_grad=False)
    x.numpy_view()[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
    w.numpy_view()[:] = np.arange(12, dtype=np.float32).reshape(3, 4)
    b.numpy_view()[:] = np.arange(4, dtype=np.float32)
    return x, w, b


def _model(x, w, b):
    return (x.matmul(w) + b).relu()


def test_forward_graph_capture_deterministic():
    x, w, b = _make_inputs()
    out1, g1 = capture_graph(_model, x, w, b)
    out2, g2 = capture_graph(_model, x, w, b)
    assert graph_signature(g1) == graph_signature(g2)
    assert out1.shape() == out2.shape()


def test_forward_graph_fusion_matmul_bias_relu():
    x, w, b = _make_inputs()
    _, graph = capture_graph(_model, x, w, b)
    plan = compile_graph(graph)
    ops = [n.get("op") for n in plan["nodes"]]
    assert "matmul_bias_relu" in ops


def test_forward_graph_cache_run():
    x, w, b = _make_inputs()
    out1, hit1 = graph_exec_cache_run(_model, x, w, b)
    out2, hit2 = graph_exec_cache_run(_model, x, w, b)
    assert hit1 is False
    assert hit2 is True
    assert np.allclose(out1.to_numpy(), out2.to_numpy())

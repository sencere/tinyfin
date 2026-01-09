import os
import sys

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor, graph_cache_key, graph_cache_clear, graph_cache_run


def _make_graph(n):
    x = Tensor.new([n], requires_grad=True)
    x.numpy_view()[:] = np.arange(n, dtype=np.float32)
    y = (x * x + x).sum()
    return y


def test_graph_cache_key_stable():
    y = _make_graph(3)
    k1 = graph_cache_key(y)
    k2 = graph_cache_key(y)
    assert k1 == k2


def test_graph_cache_key_shape_sensitive():
    k1 = graph_cache_key(_make_graph(3))
    k2 = graph_cache_key(_make_graph(4))
    assert k1 != k2


def test_graph_cache_key_dtype_device_sensitive():
    x = Tensor.new([2], requires_grad=True)
    y = (x * x).sum()
    k1 = graph_cache_key(y)

    x64 = Tensor.new([2], requires_grad=True)
    x64.set_dtype(1)
    y64 = (x64 * x64).sum()
    k2 = graph_cache_key(y64)
    assert k1 != k2

    xg = Tensor.new([2], requires_grad=True)
    xg.set_device(1)
    yg = (xg * xg).sum()
    k3 = graph_cache_key(yg)
    assert k1 != k3


def test_graph_cache_run_fallback():
    graph_cache_clear()
    called = {"n": 0}

    def make():
        called["n"] += 1
        x = Tensor.new([2], requires_grad=True)
        return (x * x).sum()

    out1, hit1 = graph_cache_run(make)
    out2, hit2 = graph_cache_run(make)

    assert hit1 is False
    assert hit2 is True
    assert out1 is out2
    assert called["n"] == 2

import sys, os, numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def test_exp_large_values_finite():
    t = Tensor.new([2], requires_grad=False)
    t.numpy_view()[:] = [100.0, 80.0]
    out = t.exp()
    arr = out.to_numpy()
    assert np.all(np.isfinite(arr))


def test_log_small_positive_finite():
    t = Tensor.new([2], requires_grad=False)
    t.numpy_view()[:] = [1e-6, 1e-4]
    out = t.log()
    arr = out.to_numpy()
    assert np.all(np.isfinite(arr))


def test_softmax_large_values_stable():
    t = Tensor.new([1, 2], requires_grad=False)
    t.numpy_view()[:] = [[1000.0, 1000.0]]
    out = t.softmax()
    arr = out.to_numpy()
    assert np.all(np.isfinite(arr))
    np.testing.assert_allclose(arr.sum(axis=1), np.ones(1), atol=1e-4)


if __name__ == "__main__":
    test_exp_large_values_finite()
    test_log_small_positive_finite()
    test_softmax_large_values_stable()
    print("[test_overflow_checks.py] PASS")

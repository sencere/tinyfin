import sys, os, math, pytest
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor, assert_finite

def test_assert_finite_ok():
    t = Tensor.new([2])
    t.numpy_view()[:] = [1.0, 2.0]
    assert_finite(t)

def test_assert_finite_raises():
    t = Tensor.new([1])
    t.numpy_view()[:] = [math.nan]
    with pytest.raises(ValueError):
        assert_finite(t)

def test_assert_finite_iterable():
    a = Tensor.new([1]); a.numpy_view()[:] = [0.0]
    b = Tensor.new([1]); b.numpy_view()[:] = [math.inf]
    with pytest.raises(ValueError):
        assert_finite([a, b])

print('[test_assert_finite.py] PASS')

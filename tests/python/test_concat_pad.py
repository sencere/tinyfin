import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

import numpy as np
import pytest
from tinyfin import Tensor


def test_concat_and_stack():
    a = Tensor.new([2, 3], requires_grad=False); a.numpy_view()[:] = np.ones((2,3), dtype=np.float32)
    b = Tensor.new([2, 3], requires_grad=False); b.numpy_view()[:] = 2*np.ones((2,3), dtype=np.float32)
    c = a.concat(b, axis=0)
    assert c.shape() == [4, 3]
    np.testing.assert_allclose(c.to_numpy(), np.concatenate([a.to_numpy(), b.to_numpy()], axis=0))

    s = a.stack(b, axis=0)
    np.testing.assert_allclose(s.to_numpy(), np.stack([a.to_numpy(), b.to_numpy()], axis=0))

def test_pad2d():
    x = Tensor.new([1, 1, 2, 2], requires_grad=False)
    x.numpy_view()[:] = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    y = x.pad2d(1, 1, value=0.0)
    assert y.shape() == [1, 1, 4, 4]
    arr = y.to_numpy()
    np.testing.assert_allclose(arr[0,0,1:3,1:3], np.array([[1,2],[3,4]], dtype=np.float32))

def test_concat_shape_error():
    a = Tensor.new([2, 3], requires_grad=False)
    b = Tensor.new([3, 3], requires_grad=False)  # mismatch on non-axis dim
    with pytest.raises(ValueError):
        _ = a.concat(b, axis=1)


def test_concat_axis_out_of_range():
    a = Tensor.new([2, 3], requires_grad=False)
    b = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        _ = a.concat(b, axis=3)


def test_stack_shape_error():
    a = Tensor.new([2, 3], requires_grad=False)
    b = Tensor.new([2, 2], requires_grad=False)
    with pytest.raises(ValueError):
        _ = a.stack(b, axis=0)


def test_stack_axis_out_of_range():
    a = Tensor.new([2, 3], requires_grad=False)
    b = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        _ = a.stack(b, axis=5)


def test_pad2d_negative_padding_raises():
    x = Tensor.new([1, 1, 2, 2], requires_grad=False)
    with pytest.raises(ValueError):
        _ = x.pad2d(-1, 0, value=0.0)


if __name__ == "__main__":
    test_concat_and_stack()
    test_pad2d()
    test_concat_shape_error()
    test_concat_axis_out_of_range()
    test_stack_shape_error()
    test_stack_axis_out_of_range()
    test_pad2d_negative_padding_raises()
    print("[test_concat_pad.py] PASS")

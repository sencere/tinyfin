import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def test_add_device_mismatch_raises():
    a = Tensor.new([1], requires_grad=True)
    b = Tensor.new([1], requires_grad=True)
    a.set_device(0)
    b.set_device(1)
    with pytest.raises(ValueError):
        _ = a + b


def test_mul_device_mismatch_raises():
    a = Tensor.new([2], requires_grad=True)
    b = Tensor.new([2], requires_grad=True)
    a.set_device(0)
    b.set_device(1)
    with pytest.raises(ValueError):
        _ = a * b


def test_matmul_shape_mismatch_raises():
    a = Tensor.new([2, 3], requires_grad=True)
    b = Tensor.new([4, 2], requires_grad=True)
    with pytest.raises(ValueError):
        _ = a.matmul(b)


def test_conv2d_shape_mismatch_raises():
    x = Tensor.new([1, 3, 8, 8], requires_grad=True)
    w = Tensor.new([4, 2, 3, 3], requires_grad=True)  # channel mismatch
    with pytest.raises(ValueError):
        _ = x.conv2d(w)


def test_div_broadcast_mismatch_raises():
    a = Tensor.new([2, 3], requires_grad=True)
    b = Tensor.new([3, 2], requires_grad=True)
    with pytest.raises(ValueError):
        _ = a / b


if __name__ == "__main__":
    pytest.main([__file__])

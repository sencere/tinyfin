import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def test_maxpool_requires_4d():
    x = Tensor.new([2, 3, 4], requires_grad=True)
    with pytest.raises(ValueError):
        x.maxpool2d(2)


def test_maxpool_kernel_must_divide():
    x = Tensor.new([1, 1, 3, 3], requires_grad=True)
    with pytest.raises(ValueError):
        x.maxpool2d(2)


def test_maxpool_kernel_positive():
    x = Tensor.new([1, 1, 2, 2], requires_grad=True)
    with pytest.raises(ValueError):
        x.maxpool2d(0)


def test_avgpool_checks_match():
    x = Tensor.new([1, 1, 3, 3], requires_grad=True)
    with pytest.raises(ValueError):
        x.avgpool2d(2)
    x2 = Tensor.new([1, 1, 2, 2], requires_grad=True)
    out = x2.avgpool2d(1)
    assert out.shape() == [1, 1, 2, 2]


def test_pool_device_mismatch():
    x = Tensor.new([1, 1, 2, 2], requires_grad=True)
    y = Tensor.new([1, 1, 2, 2], requires_grad=True)
    x.set_device(0)
    y.set_device(1)
    # pooling uses input only; ensure operations don't mix devices inadvertently
    out1 = x.maxpool2d(1)
    out2 = y.avgpool2d(1)
    assert out1.shape() == [1, 1, 2, 2]
    assert out2.shape() == [1, 1, 2, 2]


if __name__ == "__main__":
    pytest.main([__file__])

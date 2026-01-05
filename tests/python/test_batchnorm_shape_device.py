import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.nn import BatchNorm


def test_batchnorm_channel_mismatch_raises():
    x = Tensor.new([2, 3, 4, 4], requires_grad=True)
    bn = BatchNorm(4)
    with pytest.raises(ValueError):
        bn(x)


def test_batchnorm_device_mismatch_raises():
    x = Tensor.new([2, 3, 4, 4], requires_grad=True)
    bn = BatchNorm(3)
    x.set_device(0)
    bn.running_mean.set_device(1)
    with pytest.raises(ValueError):
        bn(x)


if __name__ == "__main__":
    pytest.main([__file__])

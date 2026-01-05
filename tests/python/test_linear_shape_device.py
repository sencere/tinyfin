import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.nn import Linear


def test_linear_in_features_mismatch_raises():
    lin = Linear(4, 2)
    x = Tensor.new([3, 5], requires_grad=True)
    with pytest.raises(ValueError):
        lin(x)


def test_linear_device_mismatch_raises():
    lin = Linear(4, 2)
    x = Tensor.new([3, 4], requires_grad=True)
    x.set_device(0)
    lin.weight.tensor.set_device(1)
    with pytest.raises(ValueError):
        lin(x)


if __name__ == "__main__":
    pytest.main([__file__])

import sys, os, pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def test_squeeze_dim_out_of_range():
    t = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        t.squeeze(dim=5)


def test_unsqueeze_dim_out_of_range():
    t = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        t.unsqueeze(dim=3)


def test_reshape_size_mismatch():
    t = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        t.reshape([5])

def test_permute_length_mismatch():
    t = Tensor.new([2, 3], requires_grad=False)
    with pytest.raises(ValueError):
        t.permute([0])  # missing dim


def test_permute_invalid_order():
    t = Tensor.new([2, 3, 4], requires_grad=False)
    with pytest.raises(ValueError):
        t.permute([0, 0, 1])  # duplicate dim


if __name__ == "__main__":
    pytest.main([__file__])

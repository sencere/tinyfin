import sys, os, random
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.data import Normalize, RandomHorizontalFlip, RandomCrop, Compose


def test_transforms_tensor():
    t = Tensor.new([1, 2, 4])
    t.numpy_view()[:] = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    norm = Normalize(mean=0.0, std=1.0)
    out = norm(t)
    assert out.shape() == [1, 2, 4]

    random.seed(0)
    flip = RandomHorizontalFlip(p=1.0)
    flipped = flip(t)
    assert flipped.to_numpy().tolist() == t.to_numpy()[:, :, ::-1].tolist()

    random.seed(0)
    crop = RandomCrop((2, 3), padding=0)
    cropped = crop(t)
    assert cropped.shape() == [1, 2, 3]


def test_compose():
    t = Tensor.new([1, 2, 4])
    t.numpy_view()[:] = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    trans = Compose([Normalize(0.0, 1.0)])
    out = trans(t)
    assert out.shape() == [1, 2, 4]


if __name__ == '__main__':
    test_transforms_tensor()
    test_compose()
    print("[test_transforms.py] PASS")

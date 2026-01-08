import sys, os
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.data import TensorDataset, DataLoader


def normalize(x):
    if isinstance(x, Tensor):
        return x.to_numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [normalize(i) for i in x]
    return x


def test_dataloader_workers_prefetch():
    x = Tensor.new([8, 1])
    x.numpy_view()[:] = np.arange(8, dtype=np.float32).reshape(8, 1)
    ds = TensorDataset(x)

    seq = list(DataLoader(ds, batch_size=2, shuffle=False))
    par = list(DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, prefetch_batches=2))

    assert [normalize(b) for b in seq] == [normalize(b) for b in par]


if __name__ == '__main__':
    test_dataloader_workers_prefetch()
    print("[test_dataloader_workers.py] PASS")

import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.data import TensorDataset, DataLoader

# create simple dataset
x = Tensor.new([4, 2])
x.numpy_view()[:] = [[1, 2], [3, 4], [5, 6], [7, 8]]
ds = TensorDataset(x)

# deterministic shuffling
dl1 = list(DataLoader(ds, batch_size=2, shuffle=True, seed=42))
dl2 = list(DataLoader(ds, batch_size=2, shuffle=True, seed=42))

def normalize(x):
    import numpy as np
    if isinstance(x, Tensor):
        return x.to_numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [normalize(i) for i in x]
    return x

assert [normalize(b) for b in dl1] == [normalize(b) for b in dl2]

# drop_last behavior
dl3 = list(DataLoader(ds, batch_size=3, shuffle=False, drop_last=True))
assert len(dl3) == 1

print("[test_dataloader.py] PASS")

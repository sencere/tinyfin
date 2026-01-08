import os
import sys
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt
from tinyfin.nn import LSTM, Linear

np.random.seed(0)

T, B, I, H, O = 4, 3, 5, 6, 2

lstm = LSTM(I, H)
fc = Linear(H, O)

params = [p.tensor for p in lstm.parameters()] + [p.tensor for p in fc.parameters()]
opt = SGDOpt(params, lr=1e-2)

for step in range(5):
    x = Tensor.rand(T, B, I)
    target = Tensor.rand(B, O)

    out, _ = lstm(x, return_sequences=False)
    pred = fc(out)
    diff = pred - target
    loss = (diff * diff).sum()

    loss.backward()
    opt.step()
    opt.zero_grad()

    print(f"step {step}: loss={float(loss.to_numpy().item()):.4f}")

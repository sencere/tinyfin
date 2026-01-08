"""
Minimal CPU example: train a tiny MLP on random data.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import time
from tinyfin import Tensor
from tinyfin.nn import MLP, CrossEntropyLoss
from tinyfin.optim import SGDOpt


def main():
    np.random.seed(42)
    batch, in_dim, hidden, out_dim = 64, 32, 64, 10
    x = Tensor.from_numpy(np.random.randn(batch, in_dim).astype(np.float32), requires_grad=True)
    y = Tensor.from_numpy(np.random.randint(0, out_dim, size=batch).astype(np.float32), requires_grad=False)

    model = MLP(in_dim, [hidden], out_dim)
    loss_fn = CrossEntropyLoss()
    opt = SGDOpt(model.parameters(), lr=1e-2)

    for step in range(10):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step % 2 == 0:
            print(f"[train] epoch=0 step={step} loss={loss.to_numpy().mean():.6f}")

    print("Done. Try adjusting batch/hidden sizes for quick perf checks.")


if __name__ == "__main__":
    main()

"""
Minimal CPU example: train a tiny MLP on random data.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import time
from tinyfin import Tensor, SGDOpt, cross_entropy_logits


def main():
    np.random.seed(42)
    batch, in_dim, hidden, out_dim = 64, 32, 64, 10
    x = Tensor.new([batch, in_dim], requires_grad=True)
    y = Tensor.new([batch, out_dim], requires_grad=False)
    x.numpy_view()[:] = np.random.randn(batch, in_dim).astype(np.float32)
    y.numpy_view()[:] = np.random.randn(batch, out_dim).astype(np.float32)

    w1 = Tensor.new([in_dim, hidden], requires_grad=True)
    b1 = Tensor.new([hidden], requires_grad=True)
    w2 = Tensor.new([hidden, out_dim], requires_grad=True)
    b2 = Tensor.new([out_dim], requires_grad=True)
    for p in (w1, b1, w2, b2):
        p.numpy_view()[:] = np.random.randn(*p.shape()).astype(np.float32) * 0.1

    params = [w1, b1, w2, b2]
    opt = SGDOpt(params, lr=1e-2)

    def forward(x):
        h = (x.matmul(w1) + b1).relu()
        return h.matmul(w2) + b2

    for step in range(10):
        opt.zero_grad()
        logits = forward(x)
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
        if step % 2 == 0:
            print(f"step {step}: loss={loss.to_numpy().mean():.4f}")

    print("Done. Try adjusting batch/hidden sizes for quick perf checks.")


if __name__ == "__main__":
    main()

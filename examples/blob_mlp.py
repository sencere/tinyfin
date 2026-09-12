"""
Tiny multiclass classification example.

This trains a small MLP on three synthetic 2D clusters. It is intentionally
download-free and deterministic enough to use as a quick release smoke check.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

from tinyfin import Tensor, backend_set
from tinyfin.nn import CrossEntropyLoss, MLP
from tinyfin.optim import AdamOpt


def make_blobs(samples_per_class=64, seed=7):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.2, -1.0], [1.2, -0.8], [0.0, 1.25]], dtype=np.float32)
    xs = []
    ys = []
    for label, center in enumerate(centers):
        points = center + rng.normal(0.0, 0.18, size=(samples_per_class, 2)).astype(np.float32)
        xs.append(points)
        ys.append(np.full(samples_per_class, label, dtype=np.float32))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


def accuracy(logits, targets):
    preds = logits.to_numpy().argmax(axis=1)
    return float((preds == targets.astype(np.int64)).mean())


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    np.random.seed(7)

    x_np, y_np = make_blobs()
    x = Tensor.from_numpy(x_np, requires_grad=True)
    y = Tensor.from_numpy(y_np, requires_grad=False)

    model = MLP(in_features=2, hidden_sizes=[12], out_features=3)
    loss_fn = CrossEntropyLoss()
    opt = AdamOpt(model.parameters(), lr=0.04)

    for step in range(121):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if step % 30 == 0:
            acc = accuracy(logits, y_np)
            print(
                f"[train] epoch=0 step={step} "
                f"loss={loss.item():.6f} acc={acc * 100:.2f}% backend={backend}"
            )

    final_acc = accuracy(model(x), y_np)
    if final_acc < 0.95:
        raise SystemExit(f"expected final accuracy >= 95%, got {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

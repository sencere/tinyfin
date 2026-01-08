"""
Minimal MNIST-like CNN trainer with backend toggle.

Uses synthetic data by default to avoid downloads; set USE_REAL=1 to load built-in
MNIST loader if available.
"""
import os
import sys
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', 'python'))
sys.path.insert(0, root)

from tinyfin import backend_set
from tinyfin.nn import Module, Flatten, CrossEntropyLoss
from tinyfin.optim import SGDOpt
from tinyfin.training import Trainer
from tinyfin.data import DataLoader
from tinyfin import nn
from tinyfin.utils import assert_finite


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.fc = nn.Linear(8 * 26 * 26, 10)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.flatten(y)
        return self.fc(y)


def synthetic_dataset(n=128):
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(size=(n, 1, 28, 28), dtype=np.float32)
    ys = rng.integers(0, 10, size=(n,), dtype=np.int64)
    return xs, ys

def make_loader(batch_size=16, use_real=False, device="cpu"):
    xs, ys = synthetic_dataset()
    device_id = 1 if device == "cuda" else 0
    return DataLoader.from_numpy(
        xs,
        ys.astype(np.float32),
        batch_size=batch_size,
        shuffle=True,
        device=device_id,
        requires_grad=[False, False],
    )


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    model = SimpleCNN()
    opt = SGDOpt(model.parameters(), lr=0.01, momentum=0.9)
    loader = make_loader(device=backend)

    loss_fn = CrossEntropyLoss()

    trainer = Trainer(model, loss_fn=loss_fn, optimizer=opt)
    trainer.fit(loader, epochs=1)
    for p in model.parameters():
        assert_finite(p)
    print(f"[backend_mnist_cnn] done on backend={backend}")


if __name__ == "__main__":
    main()

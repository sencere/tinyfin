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

from tinyfin import Tensor, backend_set
from tinyfin.nn import Module
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
        self.fc = nn.Linear(8 * 26 * 26, 10)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y_shape = y.shape()
        flat = 1
        for d in y_shape[1:]:
            flat *= d
        y = y.reshape([y_shape[0], flat])
        return self.fc(y)


def synthetic_dataset(n=128):
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(size=(n, 1, 28, 28), dtype=np.float32)
    ys = rng.integers(0, 10, size=(n,), dtype=np.int64)
    return xs, ys

def to_backend(t: Tensor, device_name: str):
    # devices: cpu=0, gpu=1 (placeholder for now)
    if device_name == "cuda":
        t.set_device(1)
    else:
        t.set_device(0)
    return t


def make_loader(batch_size=16, use_real=False, device="cpu"):
    xs, ys = synthetic_dataset()
    tensors = []
    for i in range(xs.shape[0]):
        t = Tensor.new([1, 28, 28], requires_grad=False)
        t.numpy_view()[:] = xs[i]
        tensors.append((to_backend(t, device), ys[i]))

    def collate(batch):
        x_batch = Tensor.new([len(batch), 1, 28, 28], requires_grad=False)
        y_np = np.array([b[1] for b in batch], dtype=np.int64)
        x_view = x_batch.numpy_view()
        for i, (t, _) in enumerate(batch):
            x_view[i, :, :, :] = t.to_numpy()
        y = Tensor.new(list(y_np.shape), requires_grad=False)
        y.numpy_view()[:] = y_np.astype(np.float32)
        return to_backend(x_batch, device), to_backend(y, device)

    return DataLoader(tensors, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    model = SimpleCNN()
    opt = SGDOpt(model.parameters(), lr=0.01, momentum=0.9)
    loader = make_loader(device=backend)

    def loss_fn(pred, target):
        # simple MSE on one-hot labels for stub
        one_hot = Tensor.new([target.shape()[0], 10], requires_grad=False)
        oh_view = one_hot.numpy_view()
        oh_view[:] = 0.0
        for i, cls in enumerate(target.to_numpy().astype(int).tolist()):
            oh_view[i, cls] = 1.0
        diff = pred - one_hot
        sq = diff * diff
        total = sq.sum()
        count = 1
        for d in sq.shape():
            count *= d
        denom = Tensor.new([1], requires_grad=False)
        if hasattr(total, "get_device") and hasattr(denom, "set_device"):
            denom.set_device(total.get_device())
        denom.numpy_view()[0] = float(count)
        return total / denom

    trainer = Trainer(model, loss_fn=loss_fn, optimizer=opt)
    for epoch in range(1):
        trainer.train_epoch(loader, epoch=epoch)
    for p in model.parameters():
        assert_finite(p)
    print(f"[backend_mnist_cnn] done on backend={backend}")


if __name__ == "__main__":
    main()

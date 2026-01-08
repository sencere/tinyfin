"""
CIFAR-like CNN training example using only tinyfin + numpy.
Reads CIFAR-10 binary batches if present (data/cifar-10-batches-bin); otherwise uses synthetic data.
Respects TINYFIN_BACKEND for backend selection.
"""
import os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from tinyfin import Tensor, backend_set
from tinyfin import nn
from tinyfin.nn import CrossEntropyLoss
from tinyfin.optim import SGDOpt
from data_utils import ensure_cifar


def maybe_load_cifar(root="data/cifar-10-batches-bin"):
    batch_files = sorted(glob.glob(os.path.join(root, "data_batch_*")))
    if not batch_files:
        try:
            ensure_cifar(root)
            batch_files = sorted(glob.glob(os.path.join(root, "data_batch_*")))
        except Exception as e:
            print(f"[cifar_cnn] download failed, staying synthetic: {e}")
            return None, None
    try:
        imgs_list = []
        labels_list = []
        for f in batch_files:
            with open(f, "rb") as fh:
                buf = np.frombuffer(fh.read(), dtype=np.uint8)
                buf = buf.reshape(-1, 3073)
                labels_list.append(buf[:, 0])
                imgs = buf[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
                imgs_list.append(imgs)
        imgs = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return imgs, labels
    except Exception as e:
        print(f"[cifar_cnn] failed to load CIFAR files, falling back to synthetic: {e}")
    return None, None


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    batch_size = 32
    steps = 200
    num_classes = 10
    rng = np.random.default_rng(0)
    imgs, labels = maybe_load_cifar()
    data_size = len(labels) if labels is not None else 50000

    # Simple CNN: conv -> pool -> flatten -> linear
    flat_dim = 16 * 15 * 15
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flat_dim, num_classes),
    )
    loss_fn = CrossEntropyLoss()
    opt = SGDOpt(model.parameters(), lr=0.05)

    correct = 0
    total = 0
    for step in range(steps):
        if imgs is not None:
            idx = rng.integers(0, data_size, size=batch_size)
            x_np = imgs[idx]
            y_idx = labels[idx]
        else:
            x_np = rng.standard_normal((batch_size, 3, 32, 32), dtype=np.float32)
            y_idx = rng.integers(0, num_classes, size=batch_size)
        x = Tensor.from_numpy(x_np, requires_grad=True)
        y = Tensor.from_numpy(y_idx.astype(np.float32), requires_grad=False)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            preds = logits.to_numpy().argmax(axis=1)
            correct += (preds == y_idx).sum()
            total += len(y_idx)
            acc = correct / total
            print(f"[train] epoch=0 step={step} loss={loss.to_numpy().mean():.6f} acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()

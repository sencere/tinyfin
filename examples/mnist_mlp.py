"""
MNIST-like MLP training example using only tinyfin + numpy.
Reads IDX files if present (data/mnist/*.ubyte); otherwise falls back to synthetic data.
Respects TINYFIN_BACKEND for backend selection.
"""
import os, sys, time, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from tinyfin import Tensor, backend_set
from tinyfin.nn import MLP, CrossEntropyLoss
from tinyfin.optim import SGDOpt
from data_utils import ensure_mnist


def load_idx_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
        data = data.reshape(num, rows * cols)
    return data


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def maybe_load_mnist(root="data/mnist"):
    img_path = os.path.join(root, "train-images-idx3-ubyte")
    lbl_path = os.path.join(root, "train-labels-idx1-ubyte")
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        try:
            imgs = load_idx_images(img_path)
            labels = load_idx_labels(lbl_path)
            return imgs, labels
        except Exception as e:
            print(f"[mnist_mlp] failed to load MNIST files, falling back to synthetic: {e}")
    # attempt download
    try:
        ensure_mnist(root)
        imgs = load_idx_images(img_path)
        labels = load_idx_labels(lbl_path)
        return imgs, labels
    except Exception as e:
        print(f"[mnist_mlp] download failed, staying synthetic: {e}")
    return None, None


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    batch_size = 128
    hidden = 256
    in_dim = 28 * 28
    out_dim = 10
    rng = np.random.default_rng(42)

    imgs, labels = maybe_load_mnist()
    data_size = len(labels) if labels is not None else 60000

    model = MLP(in_dim, [hidden], out_dim)
    loss_fn = CrossEntropyLoss()
    opt = SGDOpt(model.parameters(), lr=0.1)

    steps = 300 if imgs is None else data_size // batch_size
    correct = 0
    total = 0
    for step in range(steps):
        if imgs is not None:
            idx = rng.integers(0, data_size, size=batch_size)
            x_np = imgs[idx]
            y_idx = labels[idx]
        else:
            x_np = rng.standard_normal((batch_size, in_dim), dtype=np.float32)
            y_idx = rng.integers(0, out_dim, size=batch_size)

        x = Tensor.from_numpy(x_np, requires_grad=True)
        y = Tensor.from_numpy(y_idx.astype(np.float32), requires_grad=False)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        # accuracy
        preds = logits.to_numpy().argmax(axis=1)
        correct += (preds == y_idx).sum()
        total += len(y_idx)

        if step % 50 == 0:
            acc = correct / total
            print(f"[train] epoch=0 step={step} loss={loss.to_numpy().mean():.6f} acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()

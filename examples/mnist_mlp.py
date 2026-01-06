"""
MNIST-like MLP training example using only tinyfin + numpy.
Reads IDX files if present (data/mnist/*.ubyte); otherwise falls back to synthetic data.
Respects TINYFIN_BACKEND for backend selection.
"""
import os, sys, time, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from tinyfin import Tensor, SGDOpt, cross_entropy_logits, backend_set
from data_utils import ensure_mnist


def relu(x: Tensor) -> Tensor:
    return x.clamp_min(0.0)


def one_hot(labels, num_classes):
    arr = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    arr[np.arange(labels.shape[0]), labels] = 1.0
    t = Tensor.new(list(arr.shape), requires_grad=False)
    t.numpy_view()[:] = arr
    return t


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
    backend_set(backend.encode() if isinstance(backend, str) else backend)
    batch_size = 128
    hidden = 256
    in_dim = 28 * 28
    out_dim = 10
    rng = np.random.default_rng(42)

    imgs, labels = maybe_load_mnist()
    data_size = len(labels) if labels is not None else 60000

    w1 = Tensor.new([in_dim, hidden], requires_grad=True); w1.numpy_view()[:] = rng.standard_normal((in_dim, hidden), dtype=np.float32) * 0.02
    b1 = Tensor.new([hidden], requires_grad=True); b1.numpy_view()[:] = 0
    w2 = Tensor.new([hidden, out_dim], requires_grad=True); w2.numpy_view()[:] = rng.standard_normal((hidden, out_dim), dtype=np.float32) * 0.02
    b2 = Tensor.new([out_dim], requires_grad=True); b2.numpy_view()[:] = 0
    params = [w1, b1, w2, b2]
    opt = SGDOpt(params, lr=0.1)

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

        x = Tensor.new([batch_size, in_dim], requires_grad=True); x.numpy_view()[:] = x_np
        y = one_hot(y_idx, out_dim)

        opt.zero_grad()
        h = relu(x.matmul(w1) + b1)
        logits = h.matmul(w2) + b2
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()

        # accuracy
        preds = logits.to_numpy().argmax(axis=1)
        correct += (preds == y_idx).sum()
        total += len(y_idx)

        if step % 50 == 0:
            acc = correct / total
            print(f"step {step} loss={loss.to_numpy().mean():.4f} acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()

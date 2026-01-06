"""
CIFAR-like CNN training example using only tinyfin + numpy.
Reads CIFAR-10 binary batches if present (data/cifar-10-batches-bin); otherwise uses synthetic data.
Respects TINYFIN_BACKEND for backend selection.
"""
import os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from tinyfin import Tensor, SGDOpt, cross_entropy_logits, backend_set
from data_utils import ensure_cifar


def relu(x: Tensor) -> Tensor:
    return x.clamp_min(0.0)


def one_hot(labels, num_classes):
    arr = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    arr[np.arange(labels.shape[0]), labels] = 1.0
    t = Tensor.new(list(arr.shape), requires_grad=False)
    t.numpy_view()[:] = arr
    return t


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
    backend_set(backend.encode() if isinstance(backend, str) else backend)
    batch_size = 32
    steps = 200
    num_classes = 10
    rng = np.random.default_rng(0)
    imgs, labels = maybe_load_cifar()
    data_size = len(labels) if labels is not None else 50000

    # Simple CNN: conv -> pool -> flatten -> linear
    w1 = Tensor.new([16, 3, 3, 3], requires_grad=True); w1.numpy_view()[:] = rng.standard_normal((16, 3, 3, 3), dtype=np.float32) * 0.05
    b1 = Tensor.new([16], requires_grad=True); b1.numpy_view()[:] = 0
    # After 32x32 input, 3x3 valid conv -> 30x30, then maxpool2d(k=2) -> 15x15
    flat_dim = 16 * 15 * 15
    w2 = Tensor.new([flat_dim, num_classes], requires_grad=True); w2.numpy_view()[:] = rng.standard_normal((flat_dim, num_classes), dtype=np.float32) * 0.05
    b2 = Tensor.new([num_classes], requires_grad=True); b2.numpy_view()[:] = 0

    params = [w1, b1, w2, b2]
    opt = SGDOpt(params, lr=0.05)

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
        x = Tensor.new([batch_size, 3, 32, 32], requires_grad=True); x.numpy_view()[:] = x_np
        y = one_hot(y_idx, num_classes)

        opt.zero_grad()
        h = relu(x.conv2d(w1, b1))
        h = h.maxpool2d(2)
        h_shape = h.shape()
        flat_size = h_shape[1] * h_shape[2] * h_shape[3]
        h_flat = h.reshape([h_shape[0], flat_size])
        logits = h_flat.matmul(w2) + b2
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            preds = logits.to_numpy().argmax(axis=1)
            correct += (preds == y_idx).sum()
            total += len(y_idx)
            acc = correct / total
            print(f"step {step} loss={loss.to_numpy().mean():.4f} acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()

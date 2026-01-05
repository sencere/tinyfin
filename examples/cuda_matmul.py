"""
Simple CUDA matmul example (requires building with ENABLE_CUDA=1 and selecting backend).
Falls back to CPU if CUDA backend is unavailable.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, backend_set, backend_name


def main():
    os.environ.setdefault("TINYFIN_BACKEND", "cuda")
    backend_set("cuda")
    print(f"backend: {backend_name()}")

    m, k, n = 256, 256, 256
    a = Tensor.new([m, k], requires_grad=False)
    b = Tensor.new([k, n], requires_grad=False)
    a.set_device(1)  # DEVICE_GPU
    b.set_device(1)
    a.numpy_view()[:] = np.random.randn(m, k).astype(np.float32)
    b.numpy_view()[:] = np.random.randn(k, n).astype(np.float32)

    c = a.matmul(b)
    print("c shape:", c.shape(), "device:", c.get_device())
    print("c sum:", float(c.to_numpy().sum()))


if __name__ == "__main__":
    main()

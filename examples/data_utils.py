"""
Tiny data helpers for MNIST and CIFAR (downloads optional).
"""
import os
import urllib.request
import tarfile
import gzip
import shutil
from pathlib import Path


MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def ensure_mnist(root="data/mnist"):
    root = Path(root)
    img = root / "train-images-idx3-ubyte"
    lbl = root / "train-labels-idx1-ubyte"
    if img.exists() and lbl.exists():
        return img, lbl
    for fname, url in MNIST_URLS.items():
        gz_path = root / fname
        if not gz_path.exists():
            download(url, gz_path)
        # unpack
        with gzip.open(gz_path, "rb") as f_in, open(root / fname.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return img, lbl


def ensure_cifar(root="data/cifar-10-batches-bin"):
    root = Path(root)
    if (root / "data_batch_1").exists():
        return root
    tar_path = root.parent / "cifar-10-binary.tar.gz"
    if not tar_path.exists():
        download(CIFAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root.parent)
    return root


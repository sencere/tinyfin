"""Minimal dataset/transform/dataloader utilities for tinyfin."""
from __future__ import annotations

import random
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Any
import os
import urllib.request
import tarfile
import gzip
import shutil
from pathlib import Path
import numpy as np

from .tinyfin import Tensor


class Dataset:
    """Abstract dataset: implements __len__ and __getitem__."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class Transform:
    """Composable transform interface."""
    def __call__(self, sample):
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TensorDataset(Dataset):
    """Wrap one or more tensors (same length) into a dataset."""
    def __init__(self, *tensors: Tensor, transform: Optional[Callable] = None):
        if not tensors:
            raise ValueError("TensorDataset requires at least one Tensor")
        n = tensors[0].shape()[0]
        for t in tensors:
            if t.shape()[0] != n:
                raise ValueError("All tensors must have same first dimension")
        self.tensors = tensors
        self.transform = transform

    def __len__(self) -> int:
        return self.tensors[0].shape()[0]

    def __getitem__(self, idx: int):
        samples = []
        for t in self.tensors:
            arr = t.to_numpy()[idx]  # drop batch dim
            new_shape = list(arr.shape)
            sample = Tensor.new(new_shape, requires_grad=t.requires_grad() if hasattr(t, "requires_grad") else False)
            sample.numpy_view()[:] = arr
            try:
                sample.set_device(t.get_device())
            except Exception:
                pass
            samples.append(sample)
        item = tuple(samples)
        if len(item) == 1:
            item = item[0]
        if self.transform:
            item = self.transform(item)
        return item


def default_collate(batch: List[Any]) -> Any:
    """Simple collate: if items are Tensors of equal shape, stack on new batch dim."""
    first = batch[0]
    import numpy as np
    if isinstance(first, Tensor):
        arrs = [b.to_numpy() for b in batch]
        stacked = np.stack(arrs, axis=0)
        t = Tensor.new(list(stacked.shape))
        t.numpy_view()[:] = stacked
        return t
    if isinstance(first, (tuple, list)):
        # collate each field
        transposed = list(zip(*batch))
        return tuple(default_collate(list(items)) for items in transposed)
    return batch


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        collate_fn: Callable[[List[Any]], Any] = default_collate,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[Any]:
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(indices)
        batch: List[Any] = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# --------- simple built-in loaders (MNIST/CIFAR) ----------

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def _download(url, dest: Path):
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
            _download(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(root / fname.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return img, lbl


def load_mnist(root="data/mnist"):
    img_path, lbl_path = ensure_mnist(root)
    with open(img_path, "rb") as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">u4")
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
        data = data.reshape(num, rows * cols)
    with open(lbl_path, "rb") as f:
        magic, num = np.frombuffer(f.read(8), dtype=">u4")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    x = Tensor.new([len(data), rows * cols], requires_grad=False)
    x.numpy_view()[:] = data
    y = Tensor.new([len(labels)], requires_grad=False)
    y.numpy_view()[:] = labels
    return TensorDataset(x, y)


def ensure_cifar(root="data/cifar-10-batches-bin"):
    root = Path(root)
    if (root / "data_batch_1").exists():
        return root
    tar_path = root.parent / "cifar-10-binary.tar.gz"
    if not tar_path.exists():
        _download(CIFAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root.parent)
    return root


def load_cifar10(root="data/cifar-10-batches-bin"):
    root = ensure_cifar(root)
    imgs_list = []
    labels_list = []
    for i in range(1, 6):
        fpath = root / f"data_batch_{i}"
        if not fpath.exists():
            continue
        with open(fpath, "rb") as fh:
            buf = np.frombuffer(fh.read(), dtype=np.uint8).reshape(-1, 3073)
            labels_list.append(buf[:, 0])
            imgs = buf[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            imgs_list.append(imgs)
    if not imgs_list:
        raise RuntimeError("No CIFAR batch files found")
    imgs = np.concatenate(imgs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    x = Tensor.new([len(imgs), 3, 32, 32], requires_grad=False); x.numpy_view()[:] = imgs
    y = Tensor.new([len(labels)], requires_grad=False); y.numpy_view()[:] = labels
    return TensorDataset(x, y)

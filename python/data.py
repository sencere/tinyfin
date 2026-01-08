"""Minimal dataset/transform/dataloader utilities for tinyfin."""
from __future__ import annotations

import random
import threading
import queue
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


class Lambda(Transform):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, sample):
        return self.fn(sample)


def _tensor_from_array(arr: np.ndarray, like: Optional[Tensor] = None) -> Tensor:
    t = Tensor.new(list(arr.shape), requires_grad=False)
    t.numpy_view()[:] = arr
    if like is not None and hasattr(like, "get_device"):
        try:
            t.set_device(like.get_device())
        except Exception:
            pass
    return t


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        if isinstance(sample, Tensor):
            arr = sample.to_numpy().astype(np.float32)
            out = (arr - self.mean) / self.std
            return _tensor_from_array(out, like=sample)
        if isinstance(sample, np.ndarray):
            return (sample.astype(np.float32) - self.mean) / self.std
        return sample


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, Tensor):
            arr = sample.to_numpy()
            flipped = _flip_array(arr)
            return _tensor_from_array(flipped, like=sample)
        if isinstance(sample, np.ndarray):
            return _flip_array(sample)
        return sample


def _flip_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, ::-1]
    if arr.ndim == 3:
        return arr[:, :, ::-1]
    if arr.ndim == 4:
        return arr[:, :, :, ::-1]
    raise ValueError(f"unsupported array ndim for flip: {arr.ndim}")


class RandomCrop(Transform):
    def __init__(self, size, padding: int = 0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = int(padding)

    def __call__(self, sample):
        if isinstance(sample, Tensor):
            arr = sample.to_numpy()
            cropped = _crop_array(arr, self.size, self.padding)
            return _tensor_from_array(cropped, like=sample)
        if isinstance(sample, np.ndarray):
            return _crop_array(sample, self.size, self.padding)
        return sample


def _crop_array(arr: np.ndarray, size, padding: int) -> np.ndarray:
    if padding > 0:
        if arr.ndim == 2:
            arr = np.pad(arr, ((padding, padding), (padding, padding)), mode="constant")
        elif arr.ndim == 3:
            arr = np.pad(arr, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
        elif arr.ndim == 4:
            arr = np.pad(arr, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        else:
            raise ValueError(f"unsupported array ndim for crop: {arr.ndim}")
    h, w = size
    if arr.ndim == 2:
        H, W = arr.shape
        if H < h or W < w:
            raise ValueError("crop size larger than input")
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        return arr[top:top + h, left:left + w]
    if arr.ndim == 3:
        C, H, W = arr.shape
        if H < h or W < w:
            raise ValueError("crop size larger than input")
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        return arr[:, top:top + h, left:left + w]
    if arr.ndim == 4:
        N, C, H, W = arr.shape
        if H < h or W < w:
            raise ValueError("crop size larger than input")
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        return arr[:, :, top:top + h, left:left + w]
    raise ValueError(f"unsupported array ndim for crop: {arr.ndim}")


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

    @classmethod
    def from_numpy(cls, *arrays, requires_grad=False, device=None, transform: Optional[Callable] = None):
        if not arrays:
            raise ValueError("TensorDataset.from_numpy requires at least one array")
        if isinstance(requires_grad, (list, tuple)):
            if len(requires_grad) != len(arrays):
                raise ValueError("requires_grad length must match arrays")
            reqs = list(requires_grad)
        else:
            reqs = [bool(requires_grad)] * len(arrays)
        tensors = []
        for arr, req in zip(arrays, reqs):
            t = Tensor.from_numpy(arr, requires_grad=req, device=device)
            tensors.append(t)
        return cls(*tensors, transform=transform)

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
        if hasattr(first, "get_device") and hasattr(t, "set_device"):
            try:
                t.set_device(first.get_device())
            except Exception:
                pass
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
        num_workers: int = 0,
        prefetch_batches: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.collate_fn = collate_fn
        self.num_workers = int(num_workers)
        self.prefetch_batches = int(prefetch_batches)

    def __iter__(self) -> Iterator[Any]:
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(indices)

        batches: List[List[int]] = []
        batch_idxs: List[int] = []
        for idx in indices:
            batch_idxs.append(idx)
            if len(batch_idxs) == self.batch_size:
                batches.append(batch_idxs)
                batch_idxs = []
        if batch_idxs and not self.drop_last:
            batches.append(batch_idxs)

        if self.num_workers <= 1 and self.prefetch_batches <= 0:
            for b in batches:
                items = [self.dataset[i] for i in b]
                yield self.collate_fn(items)
            return

        n_workers = max(1, self.num_workers)
        work_q: queue.Queue = queue.Queue()
        max_prefetch = self.prefetch_batches if self.prefetch_batches > 0 else 0
        result_q: queue.Queue = queue.Queue(maxsize=max_prefetch)

        for b_id, b in enumerate(batches):
            work_q.put((b_id, b))
        for _ in range(n_workers):
            work_q.put(None)

        def worker():
            while True:
                item = work_q.get()
                if item is None:
                    break
                b_id, b = item
                try:
                    items = [self.dataset[i] for i in b]
                    batch = self.collate_fn(items)
                    result_q.put((b_id, batch, None))
                except Exception as exc:
                    result_q.put((b_id, None, exc))

        threads = []
        for _ in range(n_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        pending = {}
        next_id = 0
        for _ in range(len(batches)):
            b_id, batch, err = result_q.get()
            if err is not None:
                raise err
            pending[b_id] = batch
            while next_id in pending:
                yield pending.pop(next_id)
                next_id += 1

        for t in threads:
            t.join()

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @classmethod
    def from_numpy(
        cls,
        *arrays,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        collate_fn: Callable[[List[Any]], Any] = default_collate,
        num_workers: int = 0,
        prefetch_batches: int = 0,
        requires_grad=False,
        device=None,
        transform: Optional[Callable] = None,
    ):
        dataset = TensorDataset.from_numpy(
            *arrays,
            requires_grad=requires_grad,
            device=device,
            transform=transform,
        )
        return cls(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
        )


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

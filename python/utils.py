"""Utility helpers for tinyfin."""
from typing import Iterable, Optional, Union, Dict, Tuple, Any
import json
import os
import math
import hashlib
import numpy as np

CHECKPOINT_MAGIC = "TINYFIN_CKPT"
CHECKPOINT_VERSION = 1

from .tinyfin import assert_finite as _assert_finite
from .tinyfin import Profiler as _Profiler
from .tinyfin import Tensor

TensorLike = Union[Tensor, Iterable[Tensor]]


def assert_finite(tensors: TensorLike, msg: Optional[str] = None) -> None:
    """Raise ValueError if any tensor contains NaN or Inf."""
    _assert_finite(tensors, msg)


Profiler = _Profiler


def _tensor_state(t: Tensor) -> Dict[str, Any]:
    arr = t.to_numpy()
    checksum = hashlib.sha256(arr.tobytes()).hexdigest()
    return {
        "shape": t.shape(),
        "dtype": str(arr.dtype),
        "data": arr.flatten().tolist(),
        "requires_grad": bool(t.requires_grad()),
        "device": int(t.get_device()) if hasattr(t, "get_device") else 0,
        "checksum": checksum,
    }


def _tensor_from_state(state: Dict[str, Any]) -> Tensor:
    shape = state.get("shape")
    data = state.get("data")
    dtype = state.get("dtype")
    checksum = state.get("checksum")
    if not isinstance(shape, list) or not isinstance(data, list):
        raise ValueError("Invalid tensor state: missing shape or data")
    # For now tinyfin stores float32; reject mismatched dtype eagerly.
    if dtype is not None and str(dtype) not in ("float32", "float"):
        raise ValueError(f"Unsupported tensor dtype in checkpoint: {dtype}")
    expected_size = math.prod(shape) if shape else 0
    if len(data) != expected_size:
        raise ValueError(f"Tensor data length mismatch: expected {expected_size}, got {len(data)}")
    if checksum:
        arr_bytes = np.asarray(data, dtype=np.float32).tobytes()
        if hashlib.sha256(arr_bytes).hexdigest() != checksum:
            raise ValueError("Tensor checksum mismatch in checkpoint")
    t = Tensor.new(shape, requires_grad=bool(state.get("requires_grad", False)))
    t.numpy_view().flat[:] = state["data"]
    if "device" in state and hasattr(t, "set_device"):
        t.set_device(int(state["device"]))
    return t


def save_checkpoint(
    path: str,
    tensors: Dict[str, Tensor],
    optimizer: Any = None,
    scheduler: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a minimal versioned checkpoint (tensors + optional optimizer/scheduler state)."""
    base, _ = os.path.splitext(path)
    meta_path = base + ".ckpt.json"
    opt_path = base + ".opt.bin" if optimizer is not None else None
    sched_path = base + ".sched.json" if scheduler is not None else None

    # ensure directory exists
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)

    if optimizer is not None and hasattr(optimizer, "save_state"):
        optimizer.save_state(opt_path)
    if scheduler is not None and hasattr(scheduler, "save_state"):
        scheduler.save_state(sched_path)

    tensor_states = {name: _tensor_state(t) for name, t in tensors.items()}
    blob = {
        "magic": CHECKPOINT_MAGIC,
        "version": CHECKPOINT_VERSION,
        "tensors": tensor_states,
        "optimizer_state": opt_path if optimizer is not None else None,
        "scheduler_state": sched_path if scheduler is not None else None,
        "metadata": metadata or {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    return meta_path


def load_checkpoint(
    path: str,
    optimizer: Any = None,
    scheduler: Any = None,
    strict: bool = True,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Load tensors and optionally hydrate optimizer/scheduler from a checkpoint.
    strict=True enforces magic/version and raises on missing optimizer/scheduler state.
    """
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if blob.get("magic") != CHECKPOINT_MAGIC:
        if strict:
            raise ValueError("Unsupported checkpoint file (magic mismatch)")
        else:
            print("[tinyfin] warning: checkpoint magic mismatch; attempting best-effort load")
    if blob.get("version") != CHECKPOINT_VERSION:
        if strict:
            raise ValueError(f"Unsupported checkpoint version: {blob.get('version')}")
        else:
            print(f"[tinyfin] warning: checkpoint version {blob.get('version')} != {CHECKPOINT_VERSION}")

    tensors = {name: _tensor_from_state(state) for name, state in blob.get("tensors", {}).items()}

    if optimizer is not None:
        if blob.get("optimizer_state") and hasattr(optimizer, "load_state"):
            try:
                optimizer.load_state(blob["optimizer_state"])
            except Exception as e:
                if strict:
                    raise
                else:
                    print(f"[tinyfin] warning: failed to load optimizer state: {e}")
        elif strict:
            raise ValueError("Optimizer state path missing in checkpoint")
    if scheduler is not None:
        if blob.get("scheduler_state") and hasattr(scheduler, "load_state"):
            try:
                scheduler.load_state(blob["scheduler_state"])
            except Exception as e:
                if strict:
                    raise
                else:
                    print(f"[tinyfin] warning: failed to load scheduler state: {e}")
        elif strict:
            raise ValueError("Scheduler state path missing in checkpoint")

    return tensors, blob.get("metadata", {})


__all__ = ["assert_finite", "Profiler", "TensorLike", "save_checkpoint", "load_checkpoint"]

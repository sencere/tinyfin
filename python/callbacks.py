"""Built-in callbacks for logging and checkpointing."""
from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional

from .training import Callback


class LoggingCallback(Callback):
    def __init__(self, log_every: int = 1, sink=None):
        self.log_every = max(1, int(log_every))
        self.sink = sink if sink is not None else print

    def on_batch_end(self, step: int, loss):
        if step % self.log_every == 0:
            try:
                val = float(loss.to_numpy().item())
            except Exception:
                val = None
            self.sink({"step": step, "loss": val})


class CheckpointCallback(Callback):
    """Save model/optimizer state via user-provided save fns."""
    def __init__(self, save_dir: str, save_every: int, save_fn):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_every = max(1, int(save_every))
        self.save_fn = save_fn
        self.last_path: Optional[str] = None

    def on_batch_end(self, step: int, loss):
        if step % self.save_every == 0:
            path = os.path.join(self.save_dir, f"ckpt_step_{step}.pt")
            self.save_fn(path)
            self.last_path = path


__all__ = ["LoggingCallback", "CheckpointCallback"]

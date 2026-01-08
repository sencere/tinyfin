"""Lightweight training utilities: callbacks and a simple Trainer loop."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Any

from .tinyfin import Tensor


class Callback:
    def on_epoch_start(self, epoch: int): ...
    def on_epoch_end(self, epoch: int): ...
    def on_batch_start(self, step: int, batch: Any): ...
    def on_batch_end(self, step: int, loss: Tensor): ...


class CallbackList(Callback):
    def __init__(self, callbacks: Sequence[Callback]):
        self.callbacks = list(callbacks)

    def on_epoch_start(self, epoch: int):
        for cb in self.callbacks: cb.on_epoch_start(epoch)

    def on_epoch_end(self, epoch: int):
        for cb in self.callbacks: cb.on_epoch_end(epoch)

    def on_batch_start(self, step: int, batch: Any):
        for cb in self.callbacks: cb.on_batch_start(step, batch)

    def on_batch_end(self, step: int, loss: Tensor):
        for cb in self.callbacks: cb.on_batch_end(step, loss)


class Trainer:
    """Minimal training loop helper."""
    def __init__(self, model: Callable[[Tensor], Tensor], loss_fn: Callable, optimizer: Any, callbacks: Optional[Iterable[Callback]] = None, scheduler: Any = None, accumulate_steps: int = 1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.callbacks = CallbackList(callbacks or [])
        self.scheduler = scheduler
        self.accumulate_steps = max(1, int(accumulate_steps))

    def train_epoch(self, dataloader, epoch: int = 0) -> float:
        last_loss = 0.0
        grad_acc_counter = 0
        self.callbacks.on_epoch_start(epoch)
        for step, batch in enumerate(dataloader):
            self.callbacks.on_batch_start(step, batch)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, target = batch[0], batch[1]
            else:
                x, target = batch, None
            out = self.model(x)
            loss = self.loss_fn(out, target) if target is not None else self.loss_fn(out)
            if hasattr(loss, "requires_grad") and loss.requires_grad():
                loss.backward()
            grad_acc_counter += 1
            if grad_acc_counter >= self.accumulate_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                grad_acc_counter = 0
                if self.scheduler is not None and hasattr(self.scheduler, "step"):
                    self.scheduler.step()
            self.callbacks.on_batch_end(step, loss)
            last_loss = float(loss.to_numpy().item()) if hasattr(loss, "to_numpy") else last_loss
            print(f"[train] epoch={epoch} step={step} loss={last_loss:.6f}")

        self.callbacks.on_epoch_end(epoch)

        # flush remaining grads if dataloader length not divisible by accumulate_steps
        if grad_acc_counter > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None and hasattr(self.scheduler, "step"):
                self.scheduler.step()

        return last_loss

__all__ = ["Callback", "CallbackList", "Trainer"]

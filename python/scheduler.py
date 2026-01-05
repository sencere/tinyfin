"""Learning-rate schedulers."""
from __future__ import annotations
from typing import Callable, Any, Dict, Any as AnyType
import json
import os


class _LROpt:
    def __init__(self, opt):
        self.opt = opt
        if not hasattr(opt, "set_lr") or not hasattr(opt, "get_lr"):
            raise TypeError("optimizer must implement set_lr/get_lr")

    def set_lr(self, lr: float):
        self.opt.set_lr(lr)

    def get_lr(self) -> float:
        return self.opt.get_lr()


class StepLR:
    """Decay learning rate by gamma every step_size steps."""
    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1):
        self.opt = _LROpt(optimizer)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.last_step = 0
        self.initial_lr = self.opt.get_lr()

    def step(self):
        self.last_step += 1
        if self.last_step % self.step_size == 0:
            new_lr = self.opt.get_lr() * self.gamma
            self.opt.set_lr(new_lr)

    def get_lr(self) -> float:
        return self.opt.get_lr()

    def state_dict(self) -> Dict[str, AnyType]:
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "last_step": self.last_step,
            "initial_lr": self.initial_lr,
            "current_lr": self.opt.get_lr(),
        }

    def load_state_dict(self, state: Dict[str, AnyType]):
        self.step_size = int(state.get("step_size", self.step_size))
        self.gamma = float(state.get("gamma", self.gamma))
        self.last_step = int(state.get("last_step", 0))
        self.initial_lr = float(state.get("initial_lr", self.initial_lr))
        if "current_lr" in state:
            self.opt.set_lr(float(state["current_lr"]))

    def save_state(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)

class ExponentialLR:
    """Exponential decay: lr = lr * gamma every step."""
    def __init__(self, optimizer: Any, gamma: float):
        self.opt = _LROpt(optimizer)
        self.gamma = float(gamma)

    def step(self):
        new_lr = self.opt.get_lr() * self.gamma
        self.opt.set_lr(new_lr)

    def get_lr(self) -> float:
        return self.opt.get_lr()

    def state_dict(self) -> Dict[str, AnyType]:
        return {"gamma": self.gamma, "current_lr": self.opt.get_lr()}

    def load_state_dict(self, state: Dict[str, AnyType]):
        self.gamma = float(state.get("gamma", self.gamma))
        if "current_lr" in state:
            self.opt.set_lr(float(state["current_lr"]))

    def save_state(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)

class LinearWarmupLR:
    """Linearly increase LR to target over warmup_steps, then hold."""
    def __init__(self, optimizer: Any, warmup_steps: int, target_lr: float):
        self.opt = _LROpt(optimizer)
        self.warmup_steps = max(1, int(warmup_steps))
        self.target_lr = float(target_lr)
        self.step_idx = 0
        self.start_lr = 0.0
        self.opt.set_lr(self.start_lr)

    def step(self):
        self.step_idx += 1
        if self.step_idx <= self.warmup_steps:
            frac = self.step_idx / float(self.warmup_steps)
            new_lr = self.start_lr + frac * (self.target_lr - self.start_lr)
            self.opt.set_lr(new_lr)

    def get_lr(self) -> float:
        return self.opt.get_lr()

    def state_dict(self) -> Dict[str, AnyType]:
        return {
            "warmup_steps": self.warmup_steps,
            "target_lr": self.target_lr,
            "step_idx": self.step_idx,
            "start_lr": self.start_lr,
            "current_lr": self.opt.get_lr(),
        }

    def load_state_dict(self, state: Dict[str, AnyType]):
        self.warmup_steps = max(1, int(state.get("warmup_steps", self.warmup_steps)))
        self.target_lr = float(state.get("target_lr", self.target_lr))
        self.step_idx = int(state.get("step_idx", self.step_idx))
        self.start_lr = float(state.get("start_lr", self.start_lr))
        if "current_lr" in state:
            self.opt.set_lr(float(state["current_lr"]))

    def save_state(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)

__all__ = ["StepLR", "ExponentialLR", "LinearWarmupLR"]

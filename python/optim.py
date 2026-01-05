"""Optimizer namespace for tinyfin."""
from typing import Iterable
from .tinyfin import SGDOpt, AdamOpt, RMSPropOpt
from .tinyfin import Tensor

ParamIterable = Iterable[Tensor]

__all__ = ["SGDOpt", "AdamOpt", "RMSPropOpt", "ParamIterable"]

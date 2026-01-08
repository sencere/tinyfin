"""Optimizer namespace for tinyfin."""
from typing import Iterable
from .tinyfin import SGDOpt as _CoreSGDOpt, AdamOpt as _CoreAdamOpt, RMSPropOpt as _CoreRMSPropOpt
from .tinyfin import Tensor

ParamIterable = Iterable[Tensor]


def _unwrap_params(params):
    out = []
    for p in params:
        if isinstance(p, Tensor):
            out.append(p)
        elif hasattr(p, "tensor") and hasattr(p.tensor, "_ptr"):
            out.append(p.tensor)
        else:
            raise TypeError(f"optimizer expects Tensor or Parameter, got {type(p)}")
    return out


class SGDOpt(_CoreSGDOpt):
    def __init__(self, params, *args, **kwargs):
        super().__init__(_unwrap_params(params), *args, **kwargs)


class AdamOpt(_CoreAdamOpt):
    def __init__(self, params, *args, **kwargs):
        super().__init__(_unwrap_params(params), *args, **kwargs)


class RMSPropOpt(_CoreRMSPropOpt):
    def __init__(self, params, *args, **kwargs):
        super().__init__(_unwrap_params(params), *args, **kwargs)


__all__ = ["SGDOpt", "AdamOpt", "RMSPropOpt", "ParamIterable"]

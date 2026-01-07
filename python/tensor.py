"""Public tensor-facing API surface for tinyfin."""
from typing import Iterable, Optional, Sequence
from .tinyfin import (
    Tensor,
    no_grad,
    bce_loss,
    cross_entropy_logits,
    sparse_categorical_crossentropy,
    categorical_crossentropy,
    relu,
    assert_finite,
)

__all__ = [
    "Tensor",
    "no_grad",
    "bce_loss",
    "cross_entropy_logits",
    "sparse_categorical_crossentropy",
    "categorical_crossentropy",
    "relu",
    "assert_finite",
    "Shape",
]

Shape = Sequence[int]

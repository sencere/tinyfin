#ifndef TINYFIN_OPS_CONCAT_H
#define TINYFIN_OPS_CONCAT_H

#include "tensor.h"

Tensor *tensor_concat(Tensor *a, Tensor *b, int axis);
Tensor *tensor_stack(Tensor *a, Tensor *b, int axis);

#endif

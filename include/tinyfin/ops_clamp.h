#ifndef CTORCH_OPS_CLAMP_H
#define CTORCH_OPS_CLAMP_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_clamp(Tensor *a, float min_val, float max_val);

#endif

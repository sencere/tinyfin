#ifndef CTORCH_OPS_LAYERNORM_H
#define CTORCH_OPS_LAYERNORM_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_layernorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps);

#endif

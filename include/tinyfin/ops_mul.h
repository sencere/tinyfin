#ifndef CTORCH_OPS_MUL_H
#define CTORCH_OPS_MUL_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_mul(Tensor *a, Tensor *b);
int tensor_mul_(Tensor *a, Tensor *b);

#endif

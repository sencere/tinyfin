#ifndef CTORCH_OPS_MATMUL_H
#define CTORCH_OPS_MATMUL_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_matmul(Tensor *a, Tensor *b);

#endif

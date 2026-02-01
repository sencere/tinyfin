#ifndef CTORCH_OPS_REDUCE_H
#define CTORCH_OPS_REDUCE_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_sum(Tensor *t);
Tensor *tensor_sum_dim(Tensor *t, int dim);
Tensor *tensor_mean(Tensor *t);
Tensor *tensor_max(Tensor *t);
Tensor *tensor_min(Tensor *t);
Tensor *tensor_argmax(Tensor *t);
Tensor *tensor_argmin(Tensor *t);

#endif

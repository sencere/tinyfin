#ifndef CTORCH_OPS_SUB_H
#define CTORCH_OPS_SUB_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_sub(Tensor *a, Tensor *b);
int tensor_sub_(Tensor *a, Tensor *b);

#endif

#ifndef CTORCH_OPS_DIV_H
#define CTORCH_OPS_DIV_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_div(Tensor *a, Tensor *b);
int tensor_div_(Tensor *a, Tensor *b);

#endif

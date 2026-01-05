#ifndef CTORCH_OPS_LINEAR_H
#define CTORCH_OPS_LINEAR_H

#include "tensor.h"
#include "autograd.h"

// Linear forward (raw operation)
Tensor *linear_op_forward(Tensor *x, Tensor *w, Tensor *b);

#endif

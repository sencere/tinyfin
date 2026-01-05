#ifndef CTORCH_OPS_L1_H
#define CTORCH_OPS_L1_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_l1_loss(Tensor *pred, Tensor *target, int reduction);

#endif

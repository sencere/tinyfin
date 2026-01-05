#ifndef CTORCH_OPS_HUBER_H
#define CTORCH_OPS_HUBER_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_huber_loss(Tensor *pred, Tensor *target, float delta, int reduction);

#endif

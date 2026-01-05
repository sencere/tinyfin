#ifndef CTORCH_OPS_LOSS_H
#define CTORCH_OPS_LOSS_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target);
Tensor *tensor_softmax_autograd(Tensor *logits);
Tensor *tensor_cross_entropy(Tensor *probs, Tensor *target);

#endif

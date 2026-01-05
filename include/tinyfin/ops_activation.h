#ifndef CTORCH_OPS_ACTIVATION_H
#define CTORCH_OPS_ACTIVATION_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_relu(Tensor *x);
Tensor *tensor_sigmoid(Tensor *x);
Tensor *tensor_tanh(Tensor *x);
Tensor *tensor_leaky_relu(Tensor *x);

#endif

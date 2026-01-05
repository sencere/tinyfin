#ifndef CTORCH_OPS_CONV2D_H
#define CTORCH_OPS_CONV2D_H

#include "tensor.h"
#include "autograd.h"

/* Naive 2D convolution (stride=1, no padding) */
Tensor *tensor_conv2d(Tensor *input, Tensor *weight, Tensor *bias);

#endif

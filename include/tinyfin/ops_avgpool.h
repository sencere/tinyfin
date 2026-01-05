#ifndef CTORCH_OPS_AVGPOOL_H
#define CTORCH_OPS_AVGPOOL_H

#include "tensor.h"
#include "autograd.h"

/* AvgPool2d (kernel, stride same as kernel), no padding */
Tensor *tensor_avgpool2d(Tensor *x, int kernel_size);

/* Global average pooling across spatial dims: input [N,C,H,W] -> output [N,C]
	Autograd supported. */
Tensor *tensor_global_avgpool2d(Tensor *x);

#endif

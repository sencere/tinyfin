#ifndef CTORCH_OPS_MAXPOOL_H
#define CTORCH_OPS_MAXPOOL_H

#include "tensor.h"
#include "autograd.h"

/* MaxPool2d (kernel, stride same as kernel), no padding */
Tensor *tensor_maxpool2d(Tensor *x, int kernel_size);

#endif

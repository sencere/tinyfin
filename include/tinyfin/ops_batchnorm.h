#ifndef CTORCH_OPS_BATCHNORM_H
#define CTORCH_OPS_BATCHNORM_H

#include "tensor.h"
#include "autograd.h"

/* Per-channel BatchNorm: input [N,C,...], gamma/beta are [C] or NULL
	training: if non-zero, compute batch statistics and update running buffers
	running_mean/running_var: optional buffers (shape [C]) to update/use
	momentum: running stats momentum (typical PyTorch default 0.1)
*/
Tensor *tensor_batchnorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps, int training, Tensor *running_mean, Tensor *running_var, float momentum);

#endif

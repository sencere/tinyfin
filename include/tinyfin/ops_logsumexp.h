#ifndef TINYFIN_OPS_LOGSUMEXP_H
#define TINYFIN_OPS_LOGSUMEXP_H

#include "tensor.h"
#include "autograd.h"

/* Stable log-sum-exp reduction over all elements of the input tensor */
Tensor *tensor_logsumexp(Tensor *t);

#endif

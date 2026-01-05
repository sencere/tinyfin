#ifndef CTORCH_OPS_SOFTMAX_H
#define CTORCH_OPS_SOFTMAX_H

/* Softmax op header kept for compatibility.
	Public prototype is provided by ops_softmax_ce.c (softmax_ce implementation).
	Other modules should include ops_softmax_ce.h or ops_loss.h when needed. */
#include "tensor.h"

/* Softmax APIs */
Tensor *tensor_softmax(Tensor *logits);
Tensor *tensor_softmax_autograd(Tensor *logits);

/* Log-Softmax: stable log-softmax with autograd support */
Tensor *tensor_log_softmax(Tensor *logits);

#endif

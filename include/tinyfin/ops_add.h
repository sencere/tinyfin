#ifndef CTORCH_OPS_ADD_H
#define CTORCH_OPS_ADD_H

#include "tensor.h"
#include "autograd.h"

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *binary_op(Tensor *a, Tensor *b,
				  void (*fwd)(Tensor*, Tensor*, Tensor*),
				  void (*bwd)(AutogradNode*));
/* In-place add: modifies `a` by adding `b`. Returns 1 on success, 0 on failure.
	Currently supports only identical-shaped tensors and does not create autograd nodes. */
int tensor_add_(Tensor *a, Tensor *b);

#endif

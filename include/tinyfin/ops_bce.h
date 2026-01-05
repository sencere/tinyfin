#ifndef CTORCH_OPS_BCE_H
#define CTORCH_OPS_BCE_H

#include "tensor.h"
#include "autograd.h"

/* Binary cross entropy. If logits=1, pred are raw logits and sigmoid applied internally.
   weight: optional per-element weights (same shape as pred/target), treated as constants.
   reduction: 0=sum, 1=mean, 2=none (elementwise)
*/
Tensor *tensor_bce_loss(Tensor *pred, Tensor *target, Tensor *weight, int logits, int reduction);

#endif

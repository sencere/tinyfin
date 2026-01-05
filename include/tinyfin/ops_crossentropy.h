#ifndef CTORCH_OPS_CROSSENTROPY_H
#define CTORCH_OPS_CROSSENTROPY_H

#include "tensor.h"
#include "autograd.h"

/* CrossEntropy with logits: logits tensor of shape [N, C], target int labels shape [N]
   weight: optional class weights of shape [C], treated as constants.
   reduction: 0=sum, 1=mean, 2=none (per-example)
*/
Tensor *tensor_cross_entropy_logits(Tensor *logits, Tensor *target, Tensor *weight, int reduction);

#endif

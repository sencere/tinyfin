#ifndef CTORCH_OPS_DROPOUT_H
#define CTORCH_OPS_DROPOUT_H

#include "tensor.h"
#include "autograd.h"

/* Dropout: if training==1, randomly zero elements with probability p and scale
   remaining by 1/(1-p). If training==0, returns input (or copy) unchanged.
   Autograd supported: forward records mask tensor in grad node so backward
   multiplies incoming grad by same mask.
*/
Tensor *tensor_dropout(Tensor *x, float p, int training);

#endif

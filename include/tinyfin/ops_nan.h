#ifndef TINYFIN_OPS_NAN_H
#define TINYFIN_OPS_NAN_H

#include "tensor.h"
#include "autograd.h"

/* Return scalar tensor =1.0 if any element is NaN or Inf, else 0.0 */
Tensor *tensor_has_nan_inf(Tensor *t);

#endif

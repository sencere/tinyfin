#ifndef TINYFIN_OPS_CLAMP_MIN_H
#define TINYFIN_OPS_CLAMP_MIN_H

#include "tensor.h"
#include "autograd.h"

/* Clamp tensor values to be at least min_val (useful for epsilon safety). */
Tensor *tensor_clamp_min(Tensor *a, float min_val);

#endif

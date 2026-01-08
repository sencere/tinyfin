#ifndef TINYFIN_OPS_SLICE_H
#define TINYFIN_OPS_SLICE_H

#include "tensor.h"

Tensor *tensor_slice(Tensor *a, int axis, int start, int end);

#endif

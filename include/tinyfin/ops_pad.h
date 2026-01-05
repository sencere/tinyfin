#ifndef TINYFIN_OPS_PAD_H
#define TINYFIN_OPS_PAD_H

#include "tensor.h"

/* 2D spatial padding for NCHW tensors (constant value). */
Tensor *tensor_pad2d(Tensor *input, int pad_h, int pad_w, float value);

#endif

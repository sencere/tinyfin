#include "utils.h"
#include <stdlib.h>

int tensor_numel(const int* shape, int ndim) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

void tensor_compute_contiguous_strides(const int* shape, int ndim, int* strides) {
    int acc = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = acc;
        acc *= shape[i];
    }
}

void unravel_index(int linear, const int* shape, int ndim, int* idx_out) {
    for (int i = ndim - 1; i >= 0; i--) {
        idx_out[i] = linear % shape[i];
        linear /= shape[i];
    }
}

int tensor_offset(const Tensor* t, const int* idx) {
    int off = 0;
    for (int i = 0; i < t->ndim; i++)
        off += idx[i] * t->strides[i];
    return off;
}
#ifndef CTORCH_UTILS_H
#define CTORCH_UTILS_H

#include "tensor.h"

// number of elements
int tensor_numel(const int* shape, int ndim);

// compute contiguous strides
void tensor_compute_contiguous_strides(const int* shape, int ndim, int* strides);

// unravel a linear index into multidimensional
void unravel_index(int linear, const int* shape, int ndim, int* idx_out);

// ravel multidimensional index to linear offset (using tensor.strides)
int tensor_offset(const Tensor* t, const int* idx);
#endif

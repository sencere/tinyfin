#ifndef CTORCH_NN_H
#define CTORCH_NN_H

#include "tensor.h"

typedef struct Linear {
	Tensor *weight;
	Tensor *bias;
} Linear;

Linear *linear_create(int in_features, int out_features);
Tensor *linear_forward(Linear *l, Tensor *x);
Tensor *linear_op_forward(Tensor *x, Tensor *w, Tensor *b);

#endif

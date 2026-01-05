#include "nn.h"
#include "tensor.h"
#include "autograd.h"
#include "ops_linear.h"
#include <stdlib.h>

/* Keep layer creation and simple forwarding here; raw linear op
   implementation lives in src/ops_linear.c to avoid duplicate symbols. */

Linear *linear_create(int in_features, int out_features) {
    Linear *l = malloc(sizeof(Linear));
    if (!l) return NULL;

    int *w_shape = (int *)malloc(sizeof(int) * 2);
    if (!w_shape) { free(l); return NULL; }
    w_shape[0] = in_features;
    w_shape[1] = out_features;
    l->weight = tensor_new(2, w_shape);

    int *b_shape = (int *)malloc(sizeof(int) * 1);
    if (!b_shape) { free(w_shape); free(l); return NULL; }
    b_shape[0] = out_features;
    l->bias = tensor_new(1, b_shape);

    return l;
}

Tensor *linear_forward(Linear *l, Tensor *x) {
    return linear_op_forward(x, l->weight, l->bias);
}
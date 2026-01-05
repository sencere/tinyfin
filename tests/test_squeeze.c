#include "tensor.h"
#include "autograd.h"
#include "ops_reshape.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *shape = malloc(sizeof(int)*1);
    shape[0] = 3;
    Tensor *t = tensor_new(1, shape);
    t->data[0]=1.0f; t->data[1]=2.0f; t->data[2]=3.0f;
    t->requires_grad = 1;

    /* unsqueeze at dim 0 -> shape [1,3] */
    Tensor *u = tensor_unsqueeze(t, 0);
    assert(u->ndim == 2);
    assert(u->shape[0] == 1 && u->shape[1] == 3);

    /* squeeze dim 0 -> back to [3] */
    Tensor *s = tensor_squeeze(u, 0);
    assert(s->ndim == 1);
    assert(s->shape[0] == 3);

    /* sum and backward through views */
    Tensor *sum = tensor_sum(s);
    tensor_zero_grad(t);
    tensor_backward(sum);

    assert(t->grad->data[0] == 1.0f);
    assert(t->grad->data[1] == 1.0f);
    assert(t->grad->data[2] == 1.0f);

    tensor_free(t); tensor_free(u); tensor_free(s); tensor_free(sum); free(shape);
    printf("[test_squeeze.c] PASS\n");
    return 0;
}

#include "tensor.h"
#include "autograd.h"
#include "ops_clamp.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *shape = (int *)malloc(sizeof(int)*2);
    shape[0]=1; shape[1]=4;

    Tensor *a = tensor_new(2, shape);
    a->data[0] = -2.0f; a->data[1] = 0.5f; a->data[2] = 3.0f; a->data[3] = 5.0f;
    a->requires_grad = 1;

    Tensor *c = tensor_clamp(a, 0.0f, 3.0f);
    /* forward: [-2,0.5,3,5] -> [0,0.5,3,3] */
    assert(c->data[0] == 0.0f);
    assert(c->data[1] == 0.5f);
    assert(c->data[2] == 3.0f);
    assert(c->data[3] == 3.0f);

    Tensor *s = tensor_sum(c);
    tensor_zero_grad(a);
    tensor_backward(s);

    /* gradients: only indices 1 and 2 pass gradient (0 was clamped up, 3 clamped down) */
    assert(a->grad->data[0] == 0.0f);
    assert(a->grad->data[1] == 1.0f);
    assert(a->grad->data[2] == 1.0f);
    assert(a->grad->data[3] == 0.0f);

    tensor_free(a); tensor_free(c); tensor_free(s); free(shape);

    printf("[test_clamp.c] PASS\n");
    return 0;
}

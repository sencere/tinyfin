#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-6

int main() {
    int shape[1] = {2};

    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);

    tensor_fill(x, 1.0f);
    tensor_fill(y, 2.0f);

    x->requires_grad = 1;
    y->requires_grad = 1;

    Tensor *a = tensor_add(x,y);
    Tensor *b = tensor_mul(a,y);

    // First backward
    tensor_zero_grad(x);
    tensor_zero_grad(y);
    /* ensure output grad buffer exists before writing */
    tensor_zero_grad(b);
    for (size_t i=0;i<b->size;i++) b->grad->data[i] = 1.0f;
    tensor_backward(b);

    float grad_x_first = x->grad->data[0];
    float grad_y_first = y->grad->data[0];

    // Second backward, accumulate gradients
    for (size_t i=0;i<b->size;i++) b->grad->data[i] = 1.0f;
    tensor_backward(b);

    // Grad should accumulate
    assert(fabs(x->grad->data[0]-2*grad_x_first)<EPS);
    assert(fabs(y->grad->data[0]-2*grad_y_first)<EPS);

    tensor_free(x);
    tensor_free(y);
    tensor_free(a);
    tensor_free(b);

    printf("[test_autograd_multiple_backward.c] PASS\n");
    return 0;
}

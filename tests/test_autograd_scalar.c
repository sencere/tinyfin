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
    int shape[1] = {1};

    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);

    x->data[0] = 2.0f;
    y->data[0] = 3.0f;

    x->requires_grad = 1;
    y->requires_grad = 1;

    Tensor *z = tensor_mul(tensor_add(x,y), y); // z = (x+y)*y

    tensor_zero_grad(x);
    tensor_zero_grad(y);
    tensor_zero_grad(z);

    z->grad->data[0] = 1.0f;

    tensor_backward(z);

    // dz/dx = y = 3
    assert(fabs(x->grad->data[0]-3.0f)<EPS);
    // dz/dy = x + 2*y = 2+6=8
    assert(fabs(y->grad->data[0]-8.0f)<EPS);

    tensor_free(x);
    tensor_free(y);
    tensor_free(z);

    printf("[test_autograd_scalar.c] PASS\n");
    return 0;
}

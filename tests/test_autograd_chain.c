#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_activation.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-6

int main() {
    int shape[3] = {2,2,2};

    Tensor *x = tensor_new(3, shape);
    Tensor *y = tensor_new(3, shape);

    tensor_fill(x, 1.0f);
    tensor_fill(y, 2.0f);

    x->requires_grad = 1;
    y->requires_grad = 1;

    Tensor *a = tensor_add(x,y);        // a = x + y
    Tensor *b = tensor_mul(a,y);        // b = a * y
    Tensor *c = tensor_relu(b);         // c = relu(b)
    Tensor *d = tensor_sum(c);          // d = sum(c)

    tensor_zero_grad(x);
    tensor_zero_grad(y);
    tensor_zero_grad(a);
    tensor_zero_grad(b);
    tensor_zero_grad(c);
    tensor_zero_grad(d);

    d->grad = tensor_zeros(d->ndim, d->shape);
    d->grad->data[0] = 1.0f;

    tensor_backward(d);

    // Check gradients manually: dL/dx = dL/db * db/da * da/dx = y
    for (size_t i=0;i<x->size;i++)
        assert(fabs(x->grad->data[i]-2.0f)<EPS);

    // dL/dy = dL/db * db/dy + dL/da * da/dy = (x+y) + 2*y = 1+2+4=7? Let's verify
    for (size_t i=0;i<y->size;i++)
        assert(fabs(y->grad->data[i]-5.0f)<EPS); // x+y=3, plus y=2? final 5

    tensor_free(x);
    tensor_free(y);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);

    printf("[test_autograd_chain.c] PASS\n");
    return 0;
}

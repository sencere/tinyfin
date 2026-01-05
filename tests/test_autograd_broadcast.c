#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#define EPS 1e-6

int main() {
    // Test broadcasting: a(2x1) + b(1x3)
    int shape_a[2] = {2,1};
    int shape_b[2] = {1,3};
    
    Tensor *a = tensor_new(2, shape_a);
    Tensor *b = tensor_new(2, shape_b);

    tensor_fill(a, 2.0f);
    tensor_fill(b, 3.0f);

    a->requires_grad = 1;
    b->requires_grad = 1;

    Tensor *c = tensor_add(a,b);        // shape (2x3)
    Tensor *d = tensor_mul(c,b);        // shape (2x3)

    // Zero gradients before backward
    tensor_zero_grad(a);
    tensor_zero_grad(b);
    tensor_zero_grad(c);
    tensor_zero_grad(d);

    // Set output gradient to 1
    for (size_t i=0;i<d->size;i++) d->grad->data[i]=1.0f;

    tensor_backward(d);

    // Check shapes of gradients
    assert(a->grad->ndim==2 && a->grad->shape[0]==2 && a->grad->shape[1]==1);
    assert(b->grad->ndim==2 && b->grad->shape[0]==1 && b->grad->shape[1]==3);

    // Print gradient values for debugging
    for (int i=0;i<2;i++) {
        for (int j=0;j<1;j++) printf("a->grad[%d,%d]=%f\n", i, j, a->grad->data[i*1+j]);
    }
    for (int i=0;i<1;i++) {
        for (int j=0;j<3;j++) printf("b->grad[%d,%d]=%f\n", i, j, b->grad->data[i*3+j]);
    }

    // Verify gradient values manually (accounting for broadcasting reduction)
    // d = (a+b)*b => per-output d/d a = b, d/d b = a + 2*b
    // but when b is broadcast across rows, gradients are summed across those rows.
    // For our values (a=2, b=3) and two rows, expected:
    // a->grad entries: sum over 3 columns of b = 3+3+3 = 9
    // b->grad entries: sum over 2 rows of (a + 2*b) = 2*(2 + 6) = 16
    for (int i=0;i<2;i++)
        for (int j=0;j<1;j++)
            assert(fabs(a->grad->data[i*1+j]-9.0f)<EPS);

    for (int i=0;i<1;i++)
        for (int j=0;j<3;j++)
            assert(fabs(b->grad->data[i*3+j]-16.0f)<EPS);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);

    printf("[test_autograd_broadcast.c] PASS\n");
    return 0;
}

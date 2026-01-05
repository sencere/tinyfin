#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
#include <math.h>   

#define EPSILON 1e-6 

// Use library-provided tensor_zero_grad and tensor_backward from autograd.h

int main() {
    int *shape_2x2 = (int *)malloc(sizeof(int) * 2);
    if (!shape_2x2) return 1;
    shape_2x2[0] = 2;
    shape_2x2[1] = 2;

    Tensor *a = tensor_new(2, shape_2x2);
    Tensor *b = tensor_new(2, shape_2x2);
    
    tensor_fill(a, 2.0f);
    tensor_fill(b, 3.0f);
    a->requires_grad = 1;
    b->requires_grad = 1;

    Tensor *c = tensor_add(a, b);
    Tensor *d = tensor_mul(c, b);   // d = (a + b) * b

    printf("[debug test] after fwd: c->data[0]=%f d->data[0]=%f\n", c->data[0], d->data[0]);

    // -----------------------------------------
    // FIX: Zero ALL gradients before backward
    // -----------------------------------------
    tensor_zero_grad(a);
    tensor_zero_grad(b);
    tensor_zero_grad(c);
    tensor_zero_grad(d);

    // -----------------------------------------
    // Now set final output gradient to 1
    // -----------------------------------------
    for (size_t i = 0; i < d->size; i++)
        d->grad->data[i] = 1.0f;

    tensor_backward(d);

    // Expected:
    // L = (a + b) * b
    // dL/da = b = 3
    // dL/db = a + 2b = 2 + 6 = 8
    float expected_dL_db = a->data[0] + 2 * b->data[0];

    for (size_t i = 0; i < a->size; i++) {
        /* debug print values to help triage gradient mismatch */
        printf("a->grad[%zu]=%f b->grad[%zu]=%f a=%f b=%f expected_b=%f\n", i, a->grad->data[i], i, b->grad->data[i], a->data[i], b->data[i], expected_dL_db);
        assert(fabs(a->grad->data[i] - b->data[i]) < EPSILON);
        assert(fabs(b->grad->data[i] - expected_dL_db) < EPSILON);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    free(shape_2x2); 
    
    printf("[test_autograd.c] PASS\n");
    return 0;
}

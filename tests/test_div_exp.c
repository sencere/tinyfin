#include "tensor.h"
#include "autograd.h"
#include "ops_div.h"
#include "ops_exp.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

int main() {
    int *shape = (int *)malloc(sizeof(int)*2);
    shape[0]=2; shape[1]=3;

    Tensor *a = tensor_new(2, shape);
    Tensor *b = tensor_new(2, shape);
    tensor_fill(a, 6.0f);
    tensor_fill(b, 2.0f);
    a->requires_grad = 1;
    b->requires_grad = 1;

    Tensor *d = tensor_div(a, b);

    // forward check: all elements should be 3.0
    for (size_t i=0;i<d->size;i++) assert(d->data[i] == 3.0f);

    // backward: sum(d) -> grad a = 1/b , grad b = -a / b^2
    Tensor *s = tensor_sum(d);
    tensor_zero_grad(a); tensor_zero_grad(b);
    tensor_backward(s);

    for (size_t i=0;i<a->size;i++) {
        float expected = 1.0f / 2.0f; // 0.5
        assert(fabsf(a->grad->data[i] - expected) < 1e-6f);
    }
    for (size_t i=0;i<b->size;i++) {
        float expected = -6.0f / (2.0f*2.0f); // -1.5
        assert(fabsf(b->grad->data[i] - expected) < 1e-6f);
    }

    // Now test exp
    tensor_free(d);
    Tensor *e = tensor_exp(a); // exp(6)=~403.4288 but we only check gradient propagation
    for (size_t i=0;i<e->size;i++) assert(e->data[i] == expf(6.0f));

    Tensor *se = tensor_sum(e);
    tensor_zero_grad(a);
    tensor_backward(se);

    for (size_t i=0;i<a->size;i++) {
        float expected = expf(6.0f);
        assert(fabsf(a->grad->data[i] - expected) < 1e-5f);
    }

    tensor_free(a); tensor_free(b); tensor_free(s); tensor_free(e); tensor_free(se);
    free(shape);

    printf("[test_div_exp.c] PASS\n");
    return 0;
}

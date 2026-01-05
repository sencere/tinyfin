#include "tensor.h"
#include "autograd.h"
#include "ops_sub.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *shape = (int *)malloc(sizeof(int)*2);
    shape[0]=2; shape[1]=2;

    Tensor *a = tensor_new(2, shape);
    tensor_fill(a, 5.0f);
    a->requires_grad = 1;

    Tensor *b = tensor_new(2, shape);
    tensor_fill(b, 3.0f);
    b->requires_grad = 1;

    free(shape);

    Tensor *c = tensor_sub(a,b);

    // Check forward values
    for (size_t i=0;i<c->size;i++) assert(c->data[i] == 2.0f);

    // Backward through sum
    Tensor *s = tensor_sum(c);
    tensor_zero_grad(a);
    tensor_zero_grad(b);
    tensor_backward(s);

    // ds/da = 1 for each element -> gradient ones
    for (size_t i=0;i<a->size;i++) assert(a->grad->data[i] == 1.0f);
    // ds/db = -1 for each element
    for (size_t i=0;i<b->size;i++) assert(b->grad->data[i] == -1.0f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(s);

    printf("[test_sub.c] PASS\n");
    return 0;
}

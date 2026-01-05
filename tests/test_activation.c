#include "tensor.h"
#include "autograd.h"
#include "ops_activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define EPS 1e-6

int main() {
    int shape[1] = {5};
    Tensor *x = tensor_new(1, shape);
    x->requires_grad = 1;

    // Initialize values
    x->data[0] = -1.0f;
    x->data[1] = 0.0f;
    x->data[2] = 1.0f;
    x->data[3] = 2.0f;
    x->data[4] = -0.5f;

    // ReLU test
    Tensor *r = tensor_relu(x);
    tensor_zero_grad(x);
    for (size_t i=0;i<r->size;i++) r->grad->data[i] = 1.0f;
    tensor_backward(r);
    for (size_t i=0;i<x->size;i++) {
        float expected = (x->data[i] > 0.0f) ? 1.0f : 0.0f;
        assert(fabsf(x->grad->data[i] - expected) < EPS);
    }

    tensor_free(r);

    // Sigmoid test
    Tensor *s = tensor_sigmoid(x);
    tensor_zero_grad(x);
    for (size_t i=0;i<s->size;i++) s->grad->data[i] = 1.0f;
    tensor_backward(s);
    for (size_t i=0;i<x->size;i++) {
        float y = 1.0f / (1.0f + expf(-x->data[i]));
        float expected = y*(1.0f - y);
        assert(fabsf(x->grad->data[i] - expected) < EPS);
    }

    tensor_free(s);

    // Tanh test
    Tensor *t = tensor_tanh(x);
    tensor_zero_grad(x);
    for (size_t i=0;i<t->size;i++) t->grad->data[i] = 1.0f;
    tensor_backward(t);
    for (size_t i=0;i<x->size;i++) {
        float y = tanhf(x->data[i]);
        float expected = 1.0f - y*y;
        assert(fabsf(x->grad->data[i] - expected) < EPS);
    }

    tensor_free(t);

    // LeakyReLU test
    Tensor *l = tensor_leaky_relu(x);
    tensor_zero_grad(x);
    for (size_t i=0;i<l->size;i++) l->grad->data[i] = 1.0f;
    tensor_backward(l);
    for (size_t i=0;i<x->size;i++) {
        float alpha = 0.01f;
        float expected = (x->data[i] > 0.0f) ? 1.0f : alpha;
        assert(fabsf(x->grad->data[i] - expected) < EPS);
    }

    tensor_free(l);
    tensor_free(x);

    printf("[test_activation.c] All activation gradient tests PASS\n");
    return 0;
}

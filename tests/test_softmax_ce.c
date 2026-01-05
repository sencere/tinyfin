#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_loss.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-5

int main() {
    // Create logits tensor
    int *shape = (int *)malloc(sizeof(int));
    shape[0] = 3;
    Tensor *logits = tensor_new(1, shape);
    free(shape);

    logits->data[0] = 1.0f;
    logits->data[1] = 2.0f;
    logits->data[2] = 3.0f;
    logits->requires_grad = 1;

    // One-hot target
    shape = (int *)malloc(sizeof(int));
    shape[0] = 3;
    Tensor *target = tensor_new(1, shape);
    free(shape);
    target->data[0] = 0.0f;
    target->data[1] = 0.0f;
    target->data[2] = 1.0f;

    // Forward: softmax
    Tensor *pred = tensor_softmax_autograd(logits);

    // Forward: cross-entropy
    Tensor *loss = tensor_cross_entropy(pred, target);

    // Zero gradients
    tensor_zero_grad(logits);
    tensor_zero_grad(pred);
    tensor_zero_grad(loss);

    // Init loss gradient
    loss->grad->data[0] = 1.0f;

    // Backward
    tensor_backward(loss);

    // Check softmax CE gradient: grad = pred - target
    for (size_t i = 0; i < logits->size; i++) {
        float expected = pred->data[i] - target->data[i];
        assert(fabsf(logits->grad->data[i] - expected) < EPS);
    }

    printf("[test_softmax_ce.c] Softmax + CE autograd PASS\n");

    tensor_free(logits);
    tensor_free(target);
    tensor_free(pred);
    tensor_free(loss);

    return 0;
}

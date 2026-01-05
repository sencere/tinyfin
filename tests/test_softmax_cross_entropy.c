#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-5

// ---------- Softmax + backward ----------
typedef struct SoftmaxNode {
    AutogradNode base;
    Tensor *input;
    Tensor *output;
} SoftmaxNode;

static void softmax_backward(AutogradNode *n) {
    SoftmaxNode *snode = (SoftmaxNode *)n;
    Tensor *out = snode->output;
    Tensor *in = snode->input;

    if (!in->grad) in->grad = tensor_zeros(in->ndim, in->shape);

    // dL/dx = out * (dL/dy) - sum(out*dL/dy)*out
    float dot = 0.0f;
    for (size_t i = 0; i < in->size; i++) {
        dot += out->data[i] * out->grad->data[i];
    }
    for (size_t i = 0; i < in->size; i++) {
        float go = out->grad->data[i]; // dL/dy
        float y = out->data[i];
        in->grad->data[i] += go * y - dot * y;
    }
}

static Tensor *tensor_softmax_autograd(Tensor *x) {
    Tensor *out = tensor_new_like(x, 1);

    // Forward
    float max_val = x->data[0];
    for (size_t i = 1; i < x->size; i++)
        if (x->data[i] > max_val) max_val = x->data[i];

    float sum = 0.0f;
    for (size_t i = 0; i < x->size; i++) {
        out->data[i] = expf(x->data[i] - max_val);
        sum += out->data[i];
    }
    for (size_t i = 0; i < x->size; i++) out->data[i] /= sum;

    // Backward node
    SoftmaxNode *node = (SoftmaxNode *)malloc(sizeof(SoftmaxNode));
    node->input = x;
    node->output = out;
    node->base.out = out;
    node->base.a = x;
    node->base.b = NULL;
    node->base.backward = softmax_backward;
    node->base.n_inputs = 1;
    node->base.inputs = (Tensor **)malloc(sizeof(Tensor *));
    node->base.inputs[0] = x;
    node->base.visited = 0;
    Tensor_attach_gradients(out, (AutogradNode *)node);

    return out;
}

// ---------- Cross-entropy loss + backward ----------
typedef struct CELossNode {
    AutogradNode base;
    Tensor *pred;
    Tensor *target;
} CELossNode;

static void celoss_backward(AutogradNode *n) {
    CELossNode *cnode = (CELossNode *)n;
    Tensor *pred = cnode->pred;
    Tensor *target = cnode->target;

    if (!pred->grad) pred->grad = tensor_zeros(pred->ndim, pred->shape);

    // dL/dx = pred - target
    /* For cross-entropy loss L = -sum target * log(pred), dL/dpred = -target / pred.
       The softmax node will multiply this by the softmax Jacobian to obtain dL/dlogits. */
    for (size_t i = 0; i < pred->size; i++) {
        pred->grad->data[i] += - target->data[i] / (pred->data[i] + 1e-12f);
    }
}

static Tensor *tensor_cross_entropy(Tensor *pred, Tensor *target) {
    Tensor *loss = tensor_new(0, NULL);
    float l = 0.0f;
    for (size_t i = 0; i < pred->size; i++)
        l -= target->data[i] * logf(pred->data[i] + 1e-12f);
    loss->data[0] = l;
    loss->requires_grad = 1;
    loss->grad = tensor_zeros(0, NULL);

    CELossNode *node = (CELossNode *)malloc(sizeof(CELossNode));
    node->pred = pred;
    node->target = target;
    node->base.out = loss;
    node->base.a = pred;
    node->base.b = NULL;
    node->base.backward = celoss_backward;
    node->base.n_inputs = 1;
    node->base.inputs = (Tensor **)malloc(sizeof(Tensor *));
    node->base.inputs[0] = pred;
    node->base.visited = 0;

    Tensor_attach_gradients(loss, (AutogradNode *)node);

    return loss;
}

// ---------- TEST ----------
int main() {
    int shape[1] = {3};
    Tensor *logits = tensor_new(1, shape);
    Tensor *target = tensor_new(1, shape);

    logits->data[0] = 1.0f; logits->data[1] = 2.0f; logits->data[2] = 3.0f;
    logits->requires_grad = 1;

    target->data[0] = 0.0f; target->data[1] = 0.0f; target->data[2] = 1.0f;

    Tensor *pred = tensor_softmax_autograd(logits);
    Tensor *loss = tensor_cross_entropy(pred, target);

    // Zero all gradients
    tensor_zero_grad(logits);
    tensor_zero_grad(pred);
    tensor_zero_grad(loss);

    // Initialize final loss gradient
    loss->grad->data[0] = 1.0f;

    // Backward
    tensor_backward(loss);

    // Expected gradients: pred - target
    for (size_t i = 0; i < logits->size; i++) {
        float expected = pred->data[i] - target->data[i];
        assert(fabsf(logits->grad->data[i] - expected) < EPS);
    }

    printf("[test_softmax_cross_entropy.c] PASS\n");

    tensor_free(logits);
    tensor_free(target);
    tensor_free(pred);
    tensor_free(loss);

    return 0;
}

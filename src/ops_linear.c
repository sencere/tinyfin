#include <stdlib.h>
#include "tensor.h"
#include "autograd.h"
#include "ops_linear.h"

// ---------- Linear backward ----------
static void linear_backward(AutogradNode *n) {
    Tensor *x = n->a;
    Tensor *w = n->b;
    Tensor *b = n->bias;
    Tensor *y = n->out;

    int B = x->shape[0];
    int IN = x->shape[1];
    int OUT = w->shape[1];

    // dX
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < IN; j++) {
            float g = 0.0f;
            for (int k = 0; k < OUT; k++)
                g += y->grad->data[i*OUT + k] * w->data[j*OUT + k];
            x->grad->data[i*IN + j] += g;
        }
    }

    // dW
    for (int i = 0; i < IN; i++) {
        for (int j = 0; j < OUT; j++) {
            float g = 0.0f;
            for (int k = 0; k < B; k++)
                g += x->data[k*IN + i] * y->grad->data[k*OUT + j];
            w->grad->data[i*OUT + j] += g;
        }
    }

    // dB
    for (int j = 0; j < OUT; j++) {
        float g = 0.0f;
        for (int k = 0; k < B; k++)
            g += y->grad->data[k*OUT + j];
        b->grad->data[j] += g;
    }
}

// ---------- Linear forward ----------
Tensor *linear_op_forward(Tensor *x, Tensor *w, Tensor *b) {
    int shape[2] = { x->shape[0], w->shape[1] };
    Tensor *y = tensor_new(2, shape);

    int B = x->shape[0];
    int IN = x->shape[1];
    int OUT = w->shape[1];

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < OUT; j++) {
            float sum = 0.0f;
            for (int k = 0; k < IN; k++)
                sum += x->data[i*IN + k] * w->data[k*OUT + j];
            y->data[i*OUT + j] = sum + b->data[j];
        }
    }

    if (x->requires_grad || w->requires_grad || b->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->out = y;
        n->a = x;
        n->b = w;
        n->bias = b;
        n->backward = linear_backward;
        n->n_inputs = 3;
        n->inputs = malloc(sizeof(Tensor*) * 3);
        n->inputs[0] = (Tensor*)x;
        n->inputs[1] = (Tensor*)w;
        n->inputs[2] = (Tensor*)b;
        n->visited = 0;

        Tensor_attach_gradients(y, n);
    }

    return y;
}

#include "ops_activation.h"
#include "autograd.h"
#include "graph.h"
#include <stdlib.h>
#include <math.h>

/* ---------- ReLU ---------- */
static void relu_bwd(AutogradNode *n) {
    Tensor *x = n->a;
    Tensor *out = n->out;
    if (!x || !out || !out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    for (size_t i = 0; i < x->size; i++)
        x->grad->data[i] += (x->data[i] > 0.0f) ? out->grad->data[i] : 0.0f;
}

Tensor *tensor_relu(Tensor *x) {
    if (!x) return NULL;
    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < x->size; i++)
        out->data[i] = (x->data[i] > 0.0f) ? x->data[i] : 0.0f;

    if (x->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;

        n->out = out;
        n->a = x;
        n->b = NULL;
        n->backward = relu_bwd;
        n->n_inputs = 1;
        n->inputs = malloc(sizeof(Tensor*) * 1);
        if (!n->inputs) { free(n); return out; }
        n->inputs[0] = x;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    {
        Tensor *inputs[1] = {x};
        graph_record_op(GRAPH_OP_RELU, out, inputs, 1);
    }
    return out;
}

/* ---------- Sigmoid ---------- */
static void sigmoid_bwd(AutogradNode *n) {
    Tensor *x = n->a;
    Tensor *out = n->out;
    if (!x || !out || !out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    for (size_t i = 0; i < x->size; i++) {
        float y = out->data[i];
        x->grad->data[i] += out->grad->data[i] * y * (1.0f - y);
    }
}

Tensor *tensor_sigmoid(Tensor *x) {
    if (!x) return NULL;
    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < x->size; i++)
        out->data[i] = 1.0f / (1.0f + expf(-x->data[i]));

    if (x->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;

        n->out = out; n->a = x; n->b = NULL; n->backward = sigmoid_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*));
        if (!n->inputs) { free(n); return out; }
        n->inputs[0] = x;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

/* ---------- Tanh ---------- */
static void tanh_bwd(AutogradNode *n) {
    Tensor *x = n->a;
    Tensor *out = n->out;
    if (!x || !out || !out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    for (size_t i = 0; i < x->size; i++) {
        float y = out->data[i];
        x->grad->data[i] += out->grad->data[i] * (1.0f - y * y);
    }
}

Tensor *tensor_tanh(Tensor *x) {
    if (!x) return NULL;
    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < x->size; i++)
        out->data[i] = tanhf(x->data[i]);

    if (x->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;

        n->out = out; n->a = x; n->b = NULL; n->backward = tanh_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*));
        if (!n->inputs) { free(n); return out; }
        n->inputs[0] = x;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

/* ---------- LeakyReLU ---------- */
static void leaky_relu_bwd(AutogradNode *n) {
    Tensor *x = n->a;
    Tensor *out = n->out;
    if (!x || !out || !out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    float alpha = 0.01f;
    for (size_t i = 0; i < x->size; i++)
        x->grad->data[i] += out->grad->data[i] * ((x->data[i] > 0.0f) ? 1.0f : alpha);
}

Tensor *tensor_leaky_relu(Tensor *x) {
    if (!x) return NULL;
    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    float alpha = 0.01f;
    for (size_t i = 0; i < x->size; i++)
        out->data[i] = (x->data[i] > 0.0f) ? x->data[i] : alpha * x->data[i];

    if (x->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;

        n->out = out; n->a = x; n->b = NULL; n->backward = leaky_relu_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*));
        if (!n->inputs) { free(n); return out; }
        n->inputs[0] = x;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

#include "ops_minmax.h"
#include "autograd.h"
#include <stdlib.h>

static void minmax_bwd(AutogradNode *n) {
    Tensor *a = n->a;
    Tensor *b = n->b;
    Tensor *out = n->out;
    if (!a || !b || !out || !out->grad) return;
    if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
    if (!b->grad) b->grad = tensor_zeros(b->ndim, b->shape);
    if (!a->grad || !b->grad) return;

    int is_min = (int)(intptr_t)n->inputs[2];
    for (size_t i = 0; i < out->size; i++) {
        float av = a->data[i];
        float bv = b->data[i];
        float go = out->grad->data[i];
        if (av == bv) {
            a->grad->data[i] += 0.5f * go;
            b->grad->data[i] += 0.5f * go;
        } else if ((av < bv) == is_min) {
            a->grad->data[i] += go;
        } else {
            b->grad->data[i] += go;
        }
    }
}

static Tensor *minmax_common(Tensor *a, Tensor *b, int is_min) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }
    Tensor *out = tensor_new_like(a, a->requires_grad || b->requires_grad);
    if (!out) return NULL;
    for (size_t i = 0; i < out->size; i++) {
        float av = a->data[i];
        float bv = b->data[i];
        out->data[i] = is_min ? (av < bv ? av : bv) : (av > bv ? av : bv);
    }
    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = a;
        n->b = b;
        n->out = out;
        n->backward = minmax_bwd;
        n->n_inputs = 2;
        n->inputs = malloc(sizeof(Tensor *) * 3);
        if (n->inputs) {
            n->inputs[0] = a;
            n->inputs[1] = b;
            n->inputs[2] = (Tensor *)(intptr_t)is_min;
        }
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

Tensor *tensor_min2(Tensor *a, Tensor *b) {
    return minmax_common(a, b, 1);
}

Tensor *tensor_max2(Tensor *a, Tensor *b) {
    return minmax_common(a, b, 0);
}

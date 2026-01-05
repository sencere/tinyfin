#include "ops_dropout.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <time.h>

/* Simple per-element dropout. Uses rand(); seed expected to be set by program if needed. */
static void dropout_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    if (!out->grad) return;
    Tensor *x = n->a;
    Tensor *mask = (Tensor *)n->inputs[1];
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);
    if (!mask) return; /* if mask missing, nothing to do (shouldn't happen)
       mask is expected to contain scale (0 or 1/(1-p)) per element */
    for (size_t i = 0; i < x->size; i++) {
        x->grad->data[i] += out->grad->data[i] * mask->data[i];
    }
}

Tensor *tensor_dropout(Tensor *x, float p, int training) {
    if (!x) return NULL;
    if (!training || p <= 0.0f) {
        /* no-op: return a copy (like other ops) */
        Tensor *out = tensor_new_like(x, x->requires_grad);
        if (!out) return NULL;
        for (size_t i = 0; i < x->size; i++) out->data[i] = x->data[i];
        return out;
    }

    /* training path: create output and mask */
    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;
    Tensor *mask = tensor_new_like_autograd(x);
    if (!mask) { tensor_free(out); return NULL; }

    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x->size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        float m = (r >= p) ? scale : 0.0f;
        mask->data[i] = m;
        out->data[i] = x->data[i] * m;
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = x; n->b = NULL; n->out = out; n->backward = dropout_bwd;
        n->n_inputs = 2; n->inputs = malloc(sizeof(Tensor*) * 2);
        n->inputs[0] = x; n->inputs[1] = mask; n->visited = 0; n->hook = NULL;
        Tensor_attach_gradients(out, n);
    } else {
        /* if not tracking grad, free mask since we won't use it */
        tensor_free(mask);
    }
    return out;
}

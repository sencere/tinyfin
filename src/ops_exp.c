#include "ops_exp.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

/* Guard against overflow for float32 exponentials */
static const float EXP_MAX_INPUT = 80.0f;

static void exp_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a; if (!a) return;
    if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
    /* derivative of exp(x) is exp(x) (which equals out->data) */
    for (size_t i=0;i<out->size;i++) {
        float go = out->grad->data[i];
        float prev = a->grad->data[i];
        a->grad->data[i] = prev + go * out->data[i];
    }
}

Tensor *tensor_exp(Tensor *a) {
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) {
        float x = a->data[i];
        if (x > EXP_MAX_INPUT) x = EXP_MAX_INPUT;
        out->data[i] = expf(x);
    }
    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = a; n->b = NULL; n->out = out; n->backward = exp_bwd; n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)*1); n->inputs[0]=a; n->visited=0; Tensor_attach_gradients(out,n);
    }
    return out;
}

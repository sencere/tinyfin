#include "ops_log.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

static const float LOG_EPS = 1e-12f;

static void log_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a;
    if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
    for (size_t i=0;i<out->size;i++) {
        float x = a->data[i];
        float unclamped = x;
        if (x < LOG_EPS) x = LOG_EPS;
        float go = out->grad->data[i];
        /* Stop gradient for inputs clamped to eps to avoid huge/NaN grads. */
        float contrib = (unclamped < LOG_EPS) ? 0.0f : (go / x);
        a->grad->data[i] += contrib;
    }
}

Tensor *tensor_log(Tensor *a) {
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) {
        float x = a->data[i];
        if (x < LOG_EPS) x = LOG_EPS;
        out->data[i] = logf(x);
    }
    if (out->requires_grad) { AutogradNode *n=malloc(sizeof(*n)); n->a=a; n->b=NULL; n->out=out; n->backward=log_bwd; n->n_inputs=1; n->inputs=malloc(sizeof(Tensor*)*1); n->inputs[0]=a; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

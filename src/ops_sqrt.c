#include "ops_sqrt.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

static void sqrt_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a;
    for (size_t i=0;i<out->size;i++) { float s = out->data[i]; float go = out->grad->data[i]; a->grad->data[i] += go * 0.5f / s; }
}

Tensor *tensor_sqrt(Tensor *a) {
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) out->data[i] = sqrtf(a->data[i]);
    if (out->requires_grad) { AutogradNode *n=malloc(sizeof(*n)); n->a=a; n->b=NULL; n->out=out; n->backward=sqrt_bwd; n->n_inputs=1; n->inputs=malloc(sizeof(Tensor*)*1); n->inputs[0]=a; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

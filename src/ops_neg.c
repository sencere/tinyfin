#include "ops_neg.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

static void neg_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a;
    for (size_t i=0;i<out->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float g = tensor_get_f32_at(out->grad, i);
            float prev = tensor_get_f32_at(a->grad, i);
            tensor_set_f32_at(a->grad, i, prev - g);
        } else {
            double g = tensor_get_f64_at(out->grad, i);
            double prev = tensor_get_f64_at(a->grad, i);
            tensor_set_f64_at(a->grad, i, prev - g);
        }
    }
}

Tensor *tensor_neg(Tensor *a) {
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float v = tensor_get_f32_at(a, i);
            tensor_set_f32_at(out, i, -v);
        } else {
            double v = tensor_get_f64_at(a, i);
            tensor_set_f64_at(out, i, -v);
        }
    }
    if (out->requires_grad) { AutogradNode *n=malloc(sizeof(*n)); n->a=a; n->b=NULL; n->out=out; n->backward=neg_bwd; n->n_inputs=1; n->inputs=malloc(sizeof(Tensor*)*1); n->inputs[0]=a; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

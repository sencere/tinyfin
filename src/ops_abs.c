#include "ops_abs.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

static void abs_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a;
    for (size_t i=0;i<out->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float av = tensor_get_f32_at(a, i);
            float sign = (av > 0.0f) ? 1.0f : ((av < 0.0f) ? -1.0f : 0.0f);
            float g = tensor_get_f32_at(out->grad, i);
            float prev = tensor_get_f32_at(a->grad, i);
            tensor_set_f32_at(a->grad, i, prev + sign * g);
        } else {
            double av = tensor_get_f64_at(a, i);
            double sign = (av > 0.0) ? 1.0 : ((av < 0.0) ? -1.0 : 0.0);
            double g = tensor_get_f64_at(out->grad, i);
            double prev = tensor_get_f64_at(a->grad, i);
            tensor_set_f64_at(a->grad, i, prev + sign * g);
        }
    }
}

Tensor *tensor_abs(Tensor *a) {
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float v = tensor_get_f32_at(a, i);
            tensor_set_f32_at(out, i, fabsf(v));
        } else {
            double v = tensor_get_f64_at(a, i);
            tensor_set_f64_at(out, i, fabs(v));
        }
    }
    if (out->requires_grad) { AutogradNode *n=malloc(sizeof(*n)); n->a=a; n->b=NULL; n->out=out; n->backward=abs_bwd; n->n_inputs=1; n->inputs=malloc(sizeof(Tensor*)*1); n->inputs[0]=a; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

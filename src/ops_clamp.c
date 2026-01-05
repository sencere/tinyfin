#include "ops_clamp.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

static void clamp_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return; Tensor *a = n->a;
    float minv = *(float*)(n->inputs + 0 + 0); /* unused placeholder */
    (void)minv;
    /* propagate gradient only where input was within (min,max) */
    for (size_t i=0;i<out->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float ai = tensor_get_f32_at(a, i);
            float g = tensor_get_f32_at(out->grad, i);
            float prev = tensor_get_f32_at(a->grad, i);
            float outv = tensor_get_f32_at(out, i);
            /* if input was not clamped, outv equals ai -> pass gradient through */
            if (outv == ai) tensor_set_f32_at(a->grad, i, prev + g);
            else tensor_set_f32_at(a->grad, i, prev + 0.0f);
        } else {
            double ai = tensor_get_f64_at(a, i);
            double g = tensor_get_f64_at(out->grad, i);
            double prev = tensor_get_f64_at(a->grad, i);
            double outv = tensor_get_f64_at(out, i);
            if (outv == ai) tensor_set_f64_at(a->grad, i, prev + g);
            else tensor_set_f64_at(a->grad, i, prev + 0.0);
        }
    }
}

Tensor *tensor_clamp(Tensor *a, float min_val, float max_val) {
    if (!a) return NULL;
    Tensor *out = tensor_new_like(a, a->requires_grad); if (!out) return NULL;
    for (size_t i=0;i<a->size;i++) {
        if (out->dtype == DTYPE_FLOAT32) {
            float v = tensor_get_f32_at(a, i);
            float r = v < min_val ? min_val : (v > max_val ? max_val : v);
            tensor_set_f32_at(out, i, r);
        } else {
            double v = tensor_get_f64_at(a, i);
            double r = v < (double)min_val ? (double)min_val : (v > (double)max_val ? (double)max_val : v);
            tensor_set_f64_at(out, i, r);
        }
    }
    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;
        n->out = out; n->a = a; n->b = NULL; n->backward = clamp_bwd; n->n_inputs = 1;
        n->inputs = malloc(sizeof(Tensor*)*1); n->inputs[0] = a; n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

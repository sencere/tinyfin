#include "ops_clamp_min.h"
#include <float.h>
#include <stdlib.h>

/* Backward: pass grad where unclamped; zero where clamped. */
static void clamp_min_bwd(AutogradNode *n){
    Tensor *out = n->out;
    if (!out || !out->grad) return;
    Tensor *a = n->a;
    if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
    for (size_t i=0;i<a->size;i++){
        if (a->dtype == DTYPE_FLOAT32) {
            float ai = tensor_get_f32_at(a, i);
            float outv = tensor_get_f32_at(out, i);
            float g = tensor_get_f32_at(out->grad, i);
            float prev = tensor_get_f32_at(a->grad, i);
            if (outv == ai) tensor_set_f32_at(a->grad, i, prev + g);
            else tensor_set_f32_at(a->grad, i, prev);
        } else {
            double ai = tensor_get_f64_at(a, i);
            double outv = tensor_get_f64_at(out, i);
            double g = tensor_get_f64_at(out->grad, i);
            double prev = tensor_get_f64_at(a->grad, i);
            if (outv == ai) tensor_set_f64_at(a->grad, i, prev + g);
            else tensor_set_f64_at(a->grad, i, prev);
        }
    }
}

Tensor *tensor_clamp_min(Tensor *a, float min_val){
    if (!a) return NULL;
    Tensor *out = tensor_new_like(a, a->requires_grad);
    if (!out) return NULL;
    for (size_t i=0;i<a->size;i++){
        if (a->dtype == DTYPE_FLOAT32) {
            float v = tensor_get_f32_at(a, i);
            float r = v < min_val ? min_val : v;
            tensor_set_f32_at(out, i, r);
        } else {
            double v = tensor_get_f64_at(a, i);
            double r = v < (double)min_val ? (double)min_val : v;
            tensor_set_f64_at(out, i, r);
        }
    }
    if (out->requires_grad){
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;
        n->out = out; n->a = a; n->b = NULL; n->backward = clamp_min_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = a; n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

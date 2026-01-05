#include "ops_leakyrelu.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

static void leaky_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    if (!out || !out->grad) return;
    Tensor *a = n->a;
    float slope = *(float *)(n->inputs[1]);
    for (size_t i = 0; i < out->size; i++) {
        float ai = a->data[i];
        float go = out->grad->data[i];
        a->grad->data[i] += go * (ai >= 0 ? 1.0f : slope);
    }
}

Tensor *tensor_leakyrelu(Tensor *a, float negative_slope) {
    Tensor *out = tensor_new_like(a, a->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        float v = a->data[i];
        out->data[i] = v >= 0 ? v : negative_slope * v;
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = a; n->b = NULL; n->out = out; n->backward = leaky_bwd;
        n->n_inputs = 2; n->inputs = malloc(sizeof(Tensor*)*2);
        n->inputs[0] = a;
        /* store slope pointer in inputs[1] as a little hack */
        n->inputs[1] = (Tensor *)malloc(sizeof(float));
        *((float *)n->inputs[1]) = negative_slope;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

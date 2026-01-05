#include "ops_layernorm.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

/* LayerNorm over last dimension: for each sample normalize last dim */
static void layernorm_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    Tensor *x = n->a;
    Tensor *gamma = n->b;
    Tensor *mean = n->inputs[1];
    Tensor *var = n->inputs[2];
    float eps = *((float *)n->inputs[3]);
    if (!out->grad) return;

    int last = x->ndim - 1;
    int D = x->shape[last];
    size_t nvec = x->size / D;

    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    for (size_t v = 0; v < nvec; v++) {
        float mu = mean->data[v];
        float vv = var->data[v];
        float inv = 1.0f / sqrtf(vv + eps);
        float db = 0.0f, dg = 0.0f;
        for (int d = 0; d < D; d++) {
            size_t idx = v*D + d;
            float go = out->grad->data[idx];
            float xv = x->data[idx];
            float xn = (xv - mu) * inv;
            db += go;
            if (gamma) dg += go * xn;
        }
        for (int d = 0; d < D; d++) {
            size_t idx = v*D + d;
            float xv = x->data[idx];
            float xn = (xv - mu) * inv;
            float go = out->grad->data[idx];
            float gxn = (gamma ? gamma->data[d] : 1.0f) * go;
            float dx = (1.0f / D) * inv * (D * gxn - db - xn * dg);
            x->grad->data[idx] += dx;
        }
    }
}

Tensor *tensor_layernorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps) {
    if (x->ndim < 1) return NULL;
    int last = x->ndim - 1;
    int D = x->shape[last];
    size_t nvec = x->size / D;

    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    Tensor *mean = tensor_zeros(1, (int[]){(int)nvec});
    Tensor *var = tensor_zeros(1, (int[]){(int)nvec});

    for (size_t v = 0; v < nvec; v++) {
        float mu = 0.0f;
        for (int d = 0; d < D; d++) mu += x->data[v*D + d];
        mu /= (float)D; mean->data[v] = mu;
        float vv = 0.0f;
        for (int d = 0; d < D; d++) { float t = x->data[v*D + d] - mu; vv += t*t; }
        vv /= (float)D; var->data[v] = vv;
        float inv = 1.0f / sqrtf(vv + eps);
        for (int d = 0; d < D; d++) {
            float xn = (x->data[v*D + d] - mu) * inv;
            out->data[v*D + d] = xn * (gamma ? gamma->data[d] : 1.0f) + (beta ? beta->data[d] : 0.0f);
        }
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = x; n->b = gamma; n->out = out; n->backward = layernorm_bwd;
        n->n_inputs = 4; n->inputs = malloc(sizeof(Tensor*)*4);
        n->inputs[0] = x; n->inputs[1] = mean; n->inputs[2] = var;
        n->inputs[3] = (Tensor *)malloc(sizeof(float)); *((float *)n->inputs[3]) = eps;
        n->visited = 0; Tensor_attach_gradients(out, n);
    }
    return out;
}

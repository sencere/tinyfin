#include "ops_batchnorm.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

/* Naive per-channel batchnorm (training-mode only, no running stats)
   x: [N, C, ...]  gamma,beta: [C] or NULL
*/

static void batchnorm_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    Tensor *x = n->a;
    Tensor *gamma = n->b; /* gamma stored as b */
    Tensor *mean = n->inputs[1];
    Tensor *var = n->inputs[2];
    float eps = *((float *)n->inputs[3]);
    Tensor *beta = (n->n_inputs > 4) ? (Tensor *)n->inputs[4] : NULL;
    if (!out->grad) return;

    int N = x->shape[0];
    int C = x->shape[1];
    size_t inner = 1;
    for (int i = 2; i < x->ndim; i++) inner *= x->shape[i];
    size_t M = (size_t)N * inner; /* elements per channel */

    /* ensure grads */
    if (gamma && gamma->requires_grad && !gamma->grad) gamma->grad = tensor_zeros(gamma->ndim, gamma->shape);
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    for (int c = 0; c < C; c++) {
        float mu = mean->data[c];
        float v = var->data[c];
        float inv = 1.0f / sqrtf(v + eps);

        /* compute dbeta, dgamma */
        float db = 0.0f, dg = 0.0f;
        /* iterate over N and spatial */
        for (int n_i = 0; n_i < N; n_i++) {
            size_t base = ((size_t)n_i * C + c) * inner;
            for (size_t s = 0; s < inner; s++) {
                size_t flat = base + s;
                float go = out->grad->data[flat];
                float xv = x->data[flat];
                float xn = (xv - mu) * inv;
                db += go;
                if (gamma) dg += go * xn;
            }
        }

        if (gamma && gamma->requires_grad) gamma->grad->data[c] += dg;
        if (beta && beta->requires_grad) {
            if (!beta->grad) beta->grad = tensor_zeros(beta->ndim, beta->shape);
            beta->grad->data[c] += db;
        }

        /* compute dx (naive formula) */
        for (int n_i = 0; n_i < N; n_i++) {
            size_t base = ((size_t)n_i * C + c) * inner;
            for (size_t s = 0; s < inner; s++) {
                size_t flat = base + s;
                float xv = x->data[flat];
                float xn = (xv - mu) * inv;
                float go = out->grad->data[flat];
                float gxn = (gamma ? gamma->data[c] : 1.0f) * go;
                /* simplified batchnorm dx (not optimized) */
                float dx = (1.0f / M) * inv * (M * gxn - db - xn * (dg));
                x->grad->data[flat] += dx;
            }
        }
    }
}

Tensor *tensor_batchnorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps, int training, Tensor *running_mean, Tensor *running_var, float momentum) {
    if (x->ndim < 2) return NULL;
    int N = x->shape[0];
    int C = x->shape[1];
    size_t inner = 1;
    for (int i = 2; i < x->ndim; i++) inner *= x->shape[i];
    size_t M = (size_t)N * inner;

    Tensor *out = tensor_new_like(x, x->requires_grad);
    if (!out) return NULL;

    Tensor *mean = tensor_zeros(1, &C);
    Tensor *var = tensor_zeros(1, &C);

    if (training) {
        for (int c = 0; c < C; c++) {
            float mu = 0.0f;
            for (int n_i = 0; n_i < N; n_i++) {
                size_t base = ((size_t)n_i * C + c) * inner;
                for (size_t s = 0; s < inner; s++) {
                    mu += x->data[base + s];
                }
            }
            mu /= (float)M;
            mean->data[c] = mu;
            (void)0;
            float vv = 0.0f;
            for (int n_i = 0; n_i < N; n_i++) {
                size_t base = ((size_t)n_i * C + c) * inner;
                for (size_t s = 0; s < inner; s++) {
                    float d = x->data[base + s] - mu;
                    vv += d * d;
                }
            }
            vv /= (float)M;
            var->data[c] = vv;

            float inv = 1.0f / sqrtf(vv + eps);
            for (int n_i = 0; n_i < N; n_i++) {
                size_t base = ((size_t)n_i * C + c) * inner;
                for (size_t s = 0; s < inner; s++) {
                    float xn = (x->data[base + s] - mu) * inv;
                    float scaled = xn * (gamma ? gamma->data[c] : 1.0f) + (beta ? beta->data[c] : 0.0f);
                    out->data[base + s] = scaled;
                }
            }

            /* update running stats if provided */
            if (running_mean) {
                running_mean->data[c] = (1.0f - momentum) * running_mean->data[c] + momentum * mu;
                (void)0;
            }
            if (running_var) {
                running_var->data[c] = (1.0f - momentum) * running_var->data[c] + momentum * vv;
                (void)0;
            }
        }
    } else {
        /* evaluation mode: use running stats if provided */
        for (int c = 0; c < C; c++) {
            float mu = running_mean ? running_mean->data[c] : 0.0f;
            float vv = running_var ? running_var->data[c] : 0.0f;
            mean->data[c] = mu;
            var->data[c] = vv;
            float inv = 1.0f / sqrtf(vv + eps);
            for (int n_i = 0; n_i < N; n_i++) {
                size_t base = ((size_t)n_i * C + c) * inner;
                for (size_t s = 0; s < inner; s++) {
                    float xn = (x->data[base + s] - mu) * inv;
                    float scaled = xn * (gamma ? gamma->data[c] : 1.0f) + (beta ? beta->data[c] : 0.0f);
                    out->data[base + s] = scaled;
                }
            }
        }
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = x; n->b = gamma; n->out = out; n->backward = batchnorm_bwd;
        n->n_inputs = 5; n->inputs = malloc(sizeof(Tensor*)*5);
        n->inputs[0] = x; n->inputs[1] = mean; n->inputs[2] = var;
        /* store eps as float pointer in inputs[3] */
        n->inputs[3] = (Tensor *)malloc(sizeof(float));
        *((float *)n->inputs[3]) = eps;
        n->inputs[4] = beta; /* may be NULL */
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }

    return out;
}

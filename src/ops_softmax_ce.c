#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "autograd.h"

// ------------------------------------------------------------
// Softmax (stable)  +  Softmax Cross-Entropy (one-hot)
// ------------------------------------------------------------

// Helper: compute per-row softmax (supports 1D [C] or 2D [B x C])
static void softmax_forward(Tensor *logits, Tensor *out) {
    if (logits->ndim == 1) {
        size_t C = logits->size;

        // max for stability
        float m = logits->data[0];
        for (size_t i = 1; i < C; i++)
            if (logits->data[i] > m) m = logits->data[i];

        // exp and sum
        float s = 0.0f;
        for (size_t i = 0; i < C; i++) {
            float e = expf(logits->data[i] - m);
            out->data[i] = e;
            s += e;
        }
        float inv_s = 1.0f / s;
        // normalize
        for (size_t i = 0; i < C; i++)
            out->data[i] *= inv_s;
        return;
    }

    // 2D: [B x C] (softmax across last axis)
    size_t B = logits->shape[0];
    size_t C = logits->shape[1];

    for (size_t b = 0; b < B; b++) {
        // max for this row
        float m = logits->data[b*C + 0];
        for (size_t j = 1; j < C; j++)
            if (logits->data[b*C + j] > m) m = logits->data[b*C + j];

        // exp + sum
        float s = 0.0f;
        for (size_t j = 0; j < C; j++) {
            float e = expf(logits->data[b*C + j] - m);
            out->data[b*C + j] = e;
            s += e;
        }
        float inv_s = 1.0f / s;

        // normalize
        for (size_t j = 0; j < C; j++)
            out->data[b*C + j] *= inv_s;
    }
}

// Softmax backward (vector-Jacobian product):
// given p = softmax(x) and upstream gradient g (same shape as p):
// dx = p * (g - sum(g * p))  (per row for 2D)
static void softmax_bwd(AutogradNode *n) {
    Tensor *logits = n->a;
    Tensor *probs  = n->out;

    if (!logits->requires_grad) return;
    if (!logits->grad) logits->grad = tensor_zeros(logits->ndim, logits->shape);

    if (logits->ndim == 1) {
        size_t C = logits->size;

        float dot = 0.0f;
        for (size_t i = 0; i < C; i++)
            dot += probs->data[i] * probs->grad->data[i];

        for (size_t i = 0; i < C; i++) {
            float g = probs->grad->data[i] - dot;
            logits->grad->data[i] += probs->data[i] * g;
        }
        return;
    }

    size_t B = logits->shape[0];
    size_t C = logits->shape[1];

    for (size_t b = 0; b < B; b++) {
        float dot = 0.0f;
        for (size_t j = 0; j < C; j++)
            dot += probs->data[b*C + j] * probs->grad->data[b*C + j];

        for (size_t j = 0; j < C; j++) {
            float g = probs->grad->data[b*C + j] - dot;
            logits->grad->data[b*C + j] += probs->data[b*C + j] * g;
        }
    }
}

// Public API: softmax (autograd supports backprop)
Tensor *tensor_softmax(Tensor *logits) {
    Tensor *out = tensor_new_like(logits, logits->requires_grad);
    softmax_forward(logits, out);

    if (logits->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->out = out;
        n->a = logits;
        n->b = NULL;
        n->bias = NULL;
        n->backward = softmax_bwd;
        n->n_inputs = 1;
        n->inputs = malloc(sizeof(Tensor*));
        n->inputs[0] = (Tensor*)logits;
        n->visited = 0;

        Tensor_attach_gradients(out, n);
    }
    return out;
}

// ------------------------------------------------------------
// Softmax + Cross-Entropy (one-hot target)
// 
// Loss: L = -mean( sum_c target[c] * log(softmax(logits)[c]) )
// Backward: dL/dlogits = (probs - target) / N  (N = batch or 1)
// ------------------------------------------------------------

static void softmax_ce_fwd(Tensor *logits, Tensor *target, Tensor *out) {
    // out is scalar (1)
    out->data[0] = 0.0f;

    // compute softmax into a local tensor (same shape as logits)
    Tensor *probs = tensor_new_like_autograd(logits);
    softmax_forward(logits, probs);

    size_t N = (logits->ndim == 2) ? (size_t)logits->shape[0] : 1;
    size_t C = (logits->ndim == 2) ? (size_t)logits->shape[1] : logits->size;

    // loss = -1/N * sum_{n,c} target[n,c] * log(probs[n,c])
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            size_t idx = (logits->ndim == 2) ? (n*C + c) : c;
            float t = target->data[idx];
            if (t != 0.0f) { // sparse one-hot: only one non-zero per row
                float p = probs->data[idx];
                if (p < 1e-12f) p = 1e-12f;
                out->data[0] -= t * logf(p);
            }
        }
    }
    out->data[0] /= (float)N;

    tensor_free(probs);
}

static void softmax_ce_bwd(AutogradNode *n) {
    Tensor *logits = n->a;   // [B x C] or [C]
    Tensor *target = n->b;   // same shape (one-hot)
    Tensor *loss   = n->out; // scalar

    if (!logits->requires_grad) return;
    if (!logits->grad) logits->grad = tensor_zeros(logits->ndim, logits->shape);

    // compute probs = softmax(logits) (in-place local)
    Tensor *probs = tensor_new_like_autograd(logits);
    softmax_forward(logits, probs);

    size_t N = (logits->ndim == 2) ? (size_t)logits->shape[0] : 1;
    float scale = loss->grad->data[0] / (float)N;

    for (size_t i = 0; i < logits->size; i++) {
        float g = (probs->data[i] - target->data[i]) * scale;
        logits->grad->data[i] += g;
    }

    tensor_free(probs);
}

Tensor *tensor_softmax_cross_entropy(Tensor *logits, Tensor *target) {
    int _shape1[1] = {1};
    Tensor *out = tensor_new(1, _shape1);
    out->requires_grad = (logits->requires_grad || target->requires_grad);
    if (out->requires_grad) { out->grad = tensor_zeros(out->ndim, out->shape); if (out->grad) out->grad->requires_grad = 0; }

    softmax_ce_fwd(logits, target, out);

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->out = out;
        n->a = logits;
        n->b = target;
        n->bias = NULL;
        n->backward = softmax_ce_bwd;
        n->n_inputs = 2;
        n->inputs = malloc(sizeof(Tensor*) * 2);
        n->inputs[0] = (Tensor*)logits;
        n->inputs[1] = (Tensor*)target;
        n->visited = 0;

        Tensor_attach_gradients(out, n);
    }

    return out;
}

/* Wrapper used by tests: return softmax with autograd support */
Tensor *tensor_softmax_autograd(Tensor *logits) {
    return tensor_softmax(logits);
}

/* Cross-entropy expecting already-softmaxed probabilities as input.
 * We implement a backward that produces dL/dprobs = -(target / probs) / N
 * so that chaining through softmax backward yields the usual (probs - target)/N
 */
static void cross_entropy_probas_bwd(AutogradNode *n) {
    Tensor *probs = n->a;
    Tensor *target = n->b;
    Tensor *loss = n->out;

    if (!probs->requires_grad) return;
    if (!probs->grad) probs->grad = tensor_zeros(probs->ndim, probs->shape);

    size_t N = (probs->ndim == 2) ? (size_t)probs->shape[0] : 1;
    float scale = loss->grad->data[0] / (float)N;

    for (size_t i = 0; i < probs->size; i++) {
        float p = probs->data[i];
        float t = target->data[i];
        float g = 0.0f;
        if (p > 1e-12f) g = -(t / p) * scale;
        probs->grad->data[i] += g;
    }
}

Tensor *tensor_cross_entropy(Tensor *probs, Tensor *target) {
    int _shape1b[1] = {1};
    Tensor *out = tensor_new(1, _shape1b);
    out->requires_grad = (probs->requires_grad || target->requires_grad);
    if (out->requires_grad) { out->grad = tensor_zeros(out->ndim, out->shape); if (out->grad) out->grad->requires_grad = 0; }

    // forward using provided probabilities
    size_t N = (probs->ndim == 2) ? (size_t)probs->shape[0] : 1;
    size_t C = probs->ndim == 2 ? (size_t)probs->shape[1] : probs->size;
    out->data[0] = 0.0f;
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            size_t idx = (probs->ndim == 2) ? (n*C + c) : c;
            float p = probs->data[idx];
            float t = target->data[idx];
            if (t != 0.0f) {
                if (p < 1e-12f) p = 1e-12f;
                out->data[0] -= t * logf(p);
            }
        }
    }
    out->data[0] /= (float)N;

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->out = out;
        n->a = probs;
        n->b = target;
        n->bias = NULL;
        n->backward = cross_entropy_probas_bwd;
        n->n_inputs = 2;
        n->inputs = malloc(sizeof(Tensor*) * 2);
        n->inputs[0] = (Tensor*)probs;
        n->inputs[1] = (Tensor*)target;
        n->visited = 0;

        Tensor_attach_gradients(out, n);
    }

    return out;
}

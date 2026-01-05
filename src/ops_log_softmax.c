#include "ops_softmax.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

/* Compute log-softmax across last axis (supports 1D [C] or 2D [B x C]) */

static void log_softmax_forward(Tensor *logits, Tensor *out) {
    /* For 1D: C, for 2D: [B x C] softmax along last axis */
    if (logits->ndim == 1) {
        int C = logits->shape[0];
        float maxv = -INFINITY;
        for (int c = 0; c < C; c++) if (logits->data[c] > maxv) maxv = logits->data[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = expf(logits->data[c] - maxv);
            sum += v;
        }
        float lse = maxv + logf(sum);
        for (int c = 0; c < C; c++) out->data[c] = logits->data[c] - lse;
    } else if (logits->ndim == 2) {
        int B = logits->shape[0];
        int C = logits->shape[1];
        for (int b = 0; b < B; b++) {
            float maxv = -INFINITY;
            for (int c = 0; c < C; c++) {
                float v = logits->data[b * C + c];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (int c = 0; c < C; c++) {
                float v = expf(logits->data[b * C + c] - maxv);
                sum += v;
            }
            float lse = maxv + logf(sum);
            for (int c = 0; c < C; c++) out->data[b * C + c] = logits->data[b * C + c] - lse;
        }
    }
}

static void log_softmax_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return;
    Tensor *x = n->a; if (!x) return;

    /* out contains log_probs; p = exp(out)
       gradient: input_grad = g - p * sum(g)  (per-row for 2D) */
    if (out->ndim == 1) {
        int C = out->shape[0];
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += out->grad->data[c];
        for (int c = 0; c < C; c++) {
            float p = expf(out->data[c]);
            x->grad->data[c] += out->grad->data[c] - p * s;
        }
    } else if (out->ndim == 2) {
        int B = out->shape[0];
        int C = out->shape[1];
        for (int b = 0; b < B; b++) {
            float s = 0.0f;
            for (int c = 0; c < C; c++) s += out->grad->data[b * C + c];
            for (int c = 0; c < C; c++) {
                float p = expf(out->data[b * C + c]);
                x->grad->data[b * C + c] += out->grad->data[b * C + c] - p * s;
            }
        }
    }
}

Tensor *tensor_log_softmax(Tensor *logits) {
    if (!logits) return NULL;
    Tensor *out = tensor_new_like(logits, logits->requires_grad);
    if (!out) return NULL;
    log_softmax_forward(logits, out);
    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (!n) return out;
        n->out = out;
        n->a = logits;
        n->b = NULL;
        n->backward = log_softmax_bwd;
        n->n_inputs = 1;
        n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 1);
        n->inputs[0] = logits;
        n->visited = 0;
        n->hook = NULL;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

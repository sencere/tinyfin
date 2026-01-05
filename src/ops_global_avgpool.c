#include "ops_avgpool.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

static void global_avgpool_bwd(AutogradNode *n) {
    Tensor *out = n->out; if (!out || !out->grad) return;
    Tensor *x = n->a; if (!x) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    int N = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    float scale = 1.0f / (H * W);

    for (int ni = 0; ni < N; ni++) for (int c = 0; c < C; c++) {
        size_t oidx = (size_t)ni * C + c;
        float go = out->grad->data[oidx] * scale;
        for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) {
            size_t xidx = ((size_t)ni*C + c)*H*W + h*W + w;
            x->grad->data[xidx] += go;
        }
    }
}

Tensor *tensor_global_avgpool2d(Tensor *x) {
    if (!x || x->ndim != 4) return NULL;
    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int out_shape[2] = {N, C};
    Tensor *out = tensor_new(2, out_shape);
    if (!out) return NULL;

    float denom = (float)(H * W);
    for (int ni = 0; ni < N; ni++) for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) {
            size_t xidx = ((size_t)ni*C + c)*H*W + h*W + w;
            sum += x->data[xidx];
        }
        size_t oidx = (size_t)ni * C + c;
        out->data[oidx] = sum / denom;
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;
        n->a = x; n->b = NULL; n->out = out; n->backward = global_avgpool_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)*1);
        n->inputs[0] = x; n->visited = 0; n->hook = NULL;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

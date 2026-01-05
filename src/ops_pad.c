#include "ops_pad.h"
#include "autograd.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    AutogradNode base;
    int pad_h;
    int pad_w;
} PadCtx;

static void pad2d_bwd(AutogradNode *node) {
    PadCtx *ctx = (PadCtx *)node;
    Tensor *out = node->out;
    Tensor *x = node->a;
    if (!out || !out->grad || !x || !x->requires_grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);
    int pad_h = ctx->pad_h, pad_w = ctx->pad_w;
    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int Hout = out->shape[2], Wout = out->shape[3];
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int oh = h + pad_h;
                    int ow = w + pad_w;
                    size_t out_idx = ((size_t)n*C + c)*Hout*Wout + oh*Wout + ow;
                    size_t x_idx = ((size_t)n*C + c)*H*W + h*W + w;
                    x->grad->data[x_idx] += out->grad->data[out_idx];
                }
            }
        }
    }
}

Tensor *tensor_pad2d(Tensor *input, int pad_h, int pad_w, float value) {
    if (!input || input->ndim != 4) return NULL;
    if (pad_h < 0 || pad_w < 0) return NULL;
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int Hout = H + 2 * pad_h;
    int Wout = W + 2 * pad_w;
    int out_shape[4] = {N, C, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;
    out->requires_grad = input->requires_grad;
    out->dtype = input->dtype;
    tensor_set_dtype(out, input->dtype);
    /* fill with value */
    for (size_t i = 0; i < out->size; i++) out->data[i] = value;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int oh = h + pad_h;
                    int ow = w + pad_w;
                    size_t out_idx = ((size_t)n*C + c)*Hout*Wout + oh*Wout + ow;
                    size_t in_idx = ((size_t)n*C + c)*H*W + h*W + w;
                    out->data[out_idx] = input->data[in_idx];
                }
            }
        }
    }

    if (out->requires_grad) {
        PadCtx *ctx = (PadCtx *)malloc(sizeof(PadCtx));
        if (ctx) {
            memset(ctx, 0, sizeof(*ctx));
            ctx->base.out = out;
            ctx->base.a = input;
            ctx->base.n_inputs = 1;
            ctx->base.inputs = (Tensor **)malloc(sizeof(Tensor *));
            if (ctx->base.inputs) ctx->base.inputs[0] = input;
            ctx->base.backward = pad2d_bwd;
            ctx->pad_h = pad_h;
            ctx->pad_w = pad_w;
            Tensor_attach_gradients(out, (AutogradNode *)ctx);
        }
    }

    return out;
}

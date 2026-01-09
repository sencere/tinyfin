#include "ops_maxpool.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

/* x: [N, C, H, W] -> out [N, C, Hk, Wk] where Hk = H / k, stride=k */
static void maxpool_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    Tensor *x = n->a;
    Tensor *idx = n->inputs[1]; /* stores argmax indices as floats */
    if (!out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    int N = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int k = *((int *)n->inputs[2]);
    int Hout = out->shape[2], Wout = out->shape[3];

    for (int n_i = 0; n_i < N; n_i++) {
        for (int c = 0; c < C; c++) {
            size_t out_base = ((size_t)n_i * C + c) * Hout * Wout;
            size_t in_base = ((size_t)n_i * C + c) * H * W;
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    size_t oidx = out_base + (size_t)ho * Wout + wo;
                    int arg = (int)idx->data[oidx];
                    int ih = (arg / k) + ho * k;
                    int iw = (arg % k) + wo * k;
                    size_t xidx = in_base + (size_t)ih * W + iw;
                    x->grad->data[xidx] += out->grad->data[oidx];
                }
            }
        }
    }
}

Tensor *tensor_maxpool2d(Tensor *x, int kernel_size) {
    if (x->ndim != 4) return NULL;
    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int k = kernel_size;
    int Hout = H / k, Wout = W / k;
    int out_shape[4] = {N, C, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;

    Tensor *idx = tensor_new(4, out_shape); /* store argmax within kernel */

    for (int n_i = 0; n_i < N; n_i++) {
        for (int c = 0; c < C; c++) {
            size_t out_base = ((size_t)n_i * C + c) * Hout * Wout;
            size_t in_base = ((size_t)n_i * C + c) * H * W;
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    float best = -INFINITY;
                    int arg = 0;
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = ho * k + kh;
                            int iw = wo * k + kw;
                            size_t xidx = in_base + (size_t)ih * W + iw;
                            float v = x->data[xidx];
                            int pos = kh * k + kw;
                            if (v > best) { best = v; arg = pos; }
                        }
                    }
                    size_t oidx = out_base + (size_t)ho * Wout + wo;
                    out->data[oidx] = best;
                    idx->data[oidx] = (float)arg;
                }
            }
        }
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = x; n->b = NULL; n->out = out; n->backward = maxpool_bwd;
        n->n_inputs = 3; n->inputs = malloc(sizeof(Tensor*)*3);
        n->inputs[0] = x; n->inputs[1] = idx; n->inputs[2] = (Tensor *)malloc(sizeof(int));
        *((int *)n->inputs[2]) = k; n->visited = 0; Tensor_attach_gradients(out, n);
    }
    return out;
}

#include "ops_avgpool.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

static void avgpool_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    Tensor *x = n->a;
    if (!out->grad) return;
    if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);

    int N = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int k = *((int *)n->inputs[1]);
    int Hout = out->shape[2], Wout = out->shape[3];
    float scale = 1.0f / (float)(k * k);

    for (int n_i = 0; n_i < N; n_i++) {
        for (int c = 0; c < C; c++) {
            size_t out_base = ((size_t)n_i * C + c) * Hout * Wout;
            size_t in_base = ((size_t)n_i * C + c) * H * W;
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    size_t oidx = out_base + (size_t)ho * Wout + wo;
                    float go = out->grad->data[oidx] * scale;
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = ho * k + kh;
                            int iw = wo * k + kw;
                            size_t xidx = in_base + (size_t)ih * W + iw;
                            x->grad->data[xidx] += go;
                        }
                    }
                }
            }
        }
    }
}

Tensor *tensor_avgpool2d(Tensor *x, int kernel_size) {
    if (x->ndim != 4) return NULL;
    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int k = kernel_size;
    int Hout = H / k, Wout = W / k;
    int out_shape[4] = {N, C, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;

    float scale = 1.0f / (float)(k * k);
    for (int n_i = 0; n_i < N; n_i++) {
        for (int c = 0; c < C; c++) {
            size_t out_base = ((size_t)n_i * C + c) * Hout * Wout;
            size_t in_base = ((size_t)n_i * C + c) * H * W;
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = ho * k + kh;
                            int iw = wo * k + kw;
                            size_t xidx = in_base + (size_t)ih * W + iw;
                            sum += x->data[xidx];
                        }
                    }
                    size_t oidx = out_base + (size_t)ho * Wout + wo;
                    out->data[oidx] = sum * scale;
                }
            }
        }
    }

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = x; n->b = NULL; n->out = out; n->backward = avgpool_bwd;
        n->n_inputs = 2; n->inputs = malloc(sizeof(Tensor*)*2);
        n->inputs[0] = x; n->inputs[1] = (Tensor *)malloc(sizeof(int)); *((int *)n->inputs[1]) = k;
        n->visited = 0; Tensor_attach_gradients(out, n);
    }
    return out;
}

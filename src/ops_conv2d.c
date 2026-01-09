#include "ops_conv2d.h"
#include "tensor.h"
#include "autograd.h"
#include "backend.h"
#include "scratch.h"
#include <stdlib.h>

/* Expect input: [N, C_in, H, W]
   weight: [C_out, C_in, KH, KW]
   bias: [C_out] or NULL
   stride=1, no padding
*/

static void conv2d_bwd(AutogradNode *n) {
    Tensor *x = n->a;      // input
    Tensor *w = n->b;      // weight
    Tensor *b = n->bias;   // bias
    Tensor *y = n->out;    // output
    if (!y->grad) return;

    int N = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int C_out = w->shape[0];
    int KH = w->shape[2];
    int KW = w->shape[3];
    int Hout = y->shape[2];
    int Wout = y->shape[3];

    // grads for weight and input
    for (int n_i = 0; n_i < N; n_i++) {
        for (int oc = 0; oc < C_out; oc++) {
            for (int ic = 0; ic < C_in; ic++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        /* accumulate gradient for weight[oc,ic,kh,kw] */
                        size_t w_idx = ((size_t)oc*C_in + ic)*KH*KW + kh*KW + kw;
                        /* ensure weight grad exists and has correct dtype */
                        if (!w->grad) w->grad = tensor_zeros(w->ndim, w->shape);
                        if (w->grad) tensor_set_dtype(w->grad, w->dtype);

                        if (w->dtype == DTYPE_FLOAT32) {
                            float g = 0.0f;
                            for (int ho = 0; ho < Hout; ho++) {
                                for (int wo = 0; wo < Wout; wo++) {
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                                    size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                    g += tensor_get_f32_at(y->grad, out_idx) * tensor_get_f32_at(x, x_idx);
                                }
                            }
                            float prev = tensor_get_f32_at(w->grad, w_idx);
                            tensor_set_f32_at(w->grad, w_idx, prev + g);
                        } else {
                            double g = 0.0;
                            for (int ho = 0; ho < Hout; ho++) {
                                for (int wo = 0; wo < Wout; wo++) {
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                                    size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                    g += tensor_get_f64_at(y->grad, out_idx) * tensor_get_f64_at(x, x_idx);
                                }
                            }
                            double prev = tensor_get_f64_at(w->grad, w_idx);
                            tensor_set_f64_at(w->grad, w_idx, prev + g);
                        }
                    }
                }
            }
        }

        if (x->requires_grad) {
            /* ensure x grad exists and has correct dtype */
            if (!x->grad) x->grad = tensor_zeros(x->ndim, x->shape);
            if (x->grad) tensor_set_dtype(x->grad, x->dtype);

            for (int oc = 0; oc < C_out; oc++) {
                for (int ho = 0; ho < Hout; ho++) {
                    for (int wo = 0; wo < Wout; wo++) {
                        size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                        if (x->dtype == DTYPE_FLOAT32) {
                            float god = tensor_get_f32_at(y->grad, out_idx);
                            for (int ic = 0; ic < C_in; ic++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    for (int kw = 0; kw < KW; kw++) {
                                        int xi_h = ho + kh;
                                        int xi_w = wo + kw;
                                        size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                        size_t w_idx = ((size_t)oc*C_in + ic)*KH*KW + kh*KW + kw;
                                        float prev = tensor_get_f32_at(x->grad, x_idx);
                                        float contrib = god * tensor_get_f32_at(w, w_idx);
                                        tensor_set_f32_at(x->grad, x_idx, prev + contrib);
                                    }
                                }
                            }
                        } else {
                            double god = tensor_get_f64_at(y->grad, out_idx);
                            for (int ic = 0; ic < C_in; ic++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    for (int kw = 0; kw < KW; kw++) {
                                        int xi_h = ho + kh;
                                        int xi_w = wo + kw;
                                        size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                        size_t w_idx = ((size_t)oc*C_in + ic)*KH*KW + kh*KW + kw;
                                        double prev = tensor_get_f64_at(x->grad, x_idx);
                                        double contrib = god * tensor_get_f64_at(w, w_idx);
                                        tensor_set_f64_at(x->grad, x_idx, prev + contrib);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (b && b->requires_grad) {
        for (int oc = 0; oc < C_out; oc++) {
            float gb = 0.0f;
            for (int n_i = 0; n_i < N; n_i++) {
                for (int ho = 0; ho < Hout; ho++) {
                    for (int wo = 0; wo < Wout; wo++) {
                        size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                        gb += y->grad->data[out_idx];
                    }
                }
            }
            b->grad->data[oc] += gb;
        }
    }
}

Tensor *tensor_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    if (input->ndim != 4 || weight->ndim != 4) return NULL;
    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int C_out = weight->shape[0];
    int KH = weight->shape[2];
    int KW = weight->shape[3];
    int Hout = H - KH + 1;
    int Wout = W - KW + 1;

    /* Try backend (e.g., CUDA) first when tensors live on GPU. Backend conv2d
       currently supports inference-only; if it returns NULL we fall back to CPU. */
    if (input->device == DEVICE_GPU && weight->device == DEVICE_GPU) {
        Backend *bk = backend_get();
        if (bk && bk->conv2d) {
            Tensor *out = bk->conv2d(input, weight, bias);
            if (out) return out;
        }
    }

    int out_shape[4] = {N, C_out, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;
    out->requires_grad = (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad));
    /* inherit dtype from weight (kernel) */
    out->dtype = weight->dtype;
    tensor_set_dtype(out, weight->dtype);

    for (int n_i = 0; n_i < N; n_i++) {
        for (int oc = 0; oc < C_out; oc++) {
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    if (out->dtype == DTYPE_FLOAT32) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < C_in; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                for (int kw = 0; kw < KW; kw++) {
                                    size_t w_idx = ((size_t)oc*C_in + ic)*KH*KW + kh*KW + kw;
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                    sum += tensor_get_f32_at(input, x_idx) * tensor_get_f32_at(weight, w_idx);
                                }
                            }
                        }
                        if (bias) sum += tensor_get_f32_at(bias, oc);
                        size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                        tensor_set_f32_at(out, out_idx, sum);
                    } else {
                        double sum = 0.0;
                        for (int ic = 0; ic < C_in; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                for (int kw = 0; kw < KW; kw++) {
                                    size_t w_idx = ((size_t)oc*C_in + ic)*KH*KW + kh*KW + kw;
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t x_idx = ((size_t)n_i*C_in + ic)*H*W + xi_h*W + xi_w;
                                    sum += tensor_get_f64_at(input, x_idx) * tensor_get_f64_at(weight, w_idx);
                                }
                            }
                        }
                        if (bias) sum += tensor_get_f64_at(bias, oc);
                        size_t out_idx = ((size_t)n_i*C_out + oc)*Hout*Wout + ho*Wout + wo;
                        tensor_set_f64_at(out, out_idx, sum);
                    }
                    
                }
            }
        }
    }
    scratch_reset();

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = input; n->b = weight; n->bias = bias; n->out = out; n->backward = conv2d_bwd;
        n->n_inputs = 3; n->inputs = malloc(sizeof(Tensor*)*3); n->inputs[0]=input; n->inputs[1]=weight; n->inputs[2]=bias; n->visited=0;
        Tensor_attach_gradients(out, n);
    }

    return out;
}

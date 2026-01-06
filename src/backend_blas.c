#ifdef TINYFIN_ENABLE_BLAS
#include "backend.h"
#include "scratch.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Tensor *blas_matmul(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "[tinyfin][blas] only float32 matmul supported; falling back\n");
        return NULL;
    }
    const int M = a->shape[0];
    const int K = a->shape[1];
    const int N = b->shape[1];
    int out_shape[2] = {M, N};
    Tensor *out = tensor_new(2, out_shape);
    if (!out) return NULL;
    out->requires_grad = (a->requires_grad || b->requires_grad);
    out->device = a->device;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, a->data, a->strides[0], b->data, b->strides[0],
                0.0f, out->data, out->strides[0]);

    return out;
}

/* im2col + GEMM conv2d (float32, CPU only) */
static Tensor *blas_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    if (!input || !weight) return NULL;
    if (input->dtype != DTYPE_FLOAT32 || weight->dtype != DTYPE_FLOAT32) return NULL;
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
    if (Hout <= 0 || Wout <= 0) return NULL;

    size_t out_elems = (size_t)N * C_out * Hout * Wout;
    int out_shape[4] = {N, C_out, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;
    out->requires_grad = (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad));
    out->device = input->device;

    int Kdim = C_in * KH * KW;
    int rows = N * Hout * Wout;
    /* im2col buffer: rows x Kdim */
    float *col = (float *)scratch_alloc((size_t)rows * Kdim * sizeof(float));
    int col_fallback = 0;
    if (!col) { col = (float *)malloc((size_t)rows * Kdim * sizeof(float)); col_fallback = 1; }
    if (!col) { tensor_free(out); return NULL; }

    float *w_mat = (float *)scratch_alloc((size_t)Kdim * C_out * sizeof(float));
    int w_fallback = 0;
    if (!w_mat) { w_mat = (float *)malloc((size_t)Kdim * C_out * sizeof(float)); w_fallback = 1; }
    if (!w_mat) { if (col_fallback) free(col); tensor_free(out); return NULL; }

    /* reshape weight to Kdim x C_out */
    for (int oc = 0; oc < C_out; oc++) {
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int k = ((ic * KH + kh) * KW) + kw;
                    size_t w_idx = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
                    w_mat[k * C_out + oc] = weight->data[w_idx];
                }
            }
        }
    }

    /* im2col */
    for (int n = 0; n < N; n++) {
        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                int row = (n * Hout + ho) * Wout + wo;
                int idx = 0;
                for (int ic = 0; ic < C_in; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = ho + kh;
                            int iw = wo + kw;
                            size_t x_idx = ((size_t)n * C_in + ic) * H * W + ih * W + iw;
                            col[row * Kdim + idx++] = input->data[x_idx];
                        }
                    }
                }
            }
        }
    }

    /* GEMM: (rows x Kdim) * (Kdim x C_out) = rows x C_out */
    float *out_mat = out->data;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, C_out, Kdim,
                1.0f, col, Kdim,
                w_mat, C_out,
                0.0f, out_mat, C_out);

    /* add bias */
    if (bias) {
        for (int r = 0; r < rows; r++) {
            for (int oc = 0; oc < C_out; oc++) {
                out_mat[r * C_out + oc] += bias->data[oc];
            }
        }
    }

    if (col_fallback) free(col);
    if (w_fallback) free(w_mat);
    scratch_reset();
    return out;
}

static Backend blas_backend = {
    .name = "blas",
    .matmul = blas_matmul,
    .conv2d = blas_conv2d,
    .add = NULL,
    .mul = NULL,
};

__attribute__((constructor))
static void register_blas_backend(void) {
    backend_register(&blas_backend);
}
#endif /* TINYFIN_ENABLE_BLAS */

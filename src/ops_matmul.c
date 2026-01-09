#include "ops_matmul.h"
#include "tensor.h"
#include "autograd.h"
#include "profiler.h"
#include "backend.h"
#include "scratch.h"
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Simple, stride-aware 2D matrix multiply (C order):
 * out = a @ b
 * where a is (M x K) and b is (K x N) producing (M x N)
 * Backward:
 *   dA = dOut @ B^T
 *   dB = A^T @ dOut
 */

static void matmul_bwd(AutogradNode *n) {
    if (!n) return;
    Tensor *a = n->a;
    Tensor *b = n->b;
    Tensor *out = n->out;
    if (!a || !b || !out || !out->grad) return;

    const int M = a->shape[0];
    const int K = a->shape[1];
    const int N = b->shape[1];

    const int a_s0 = a->strides[0];
    const int a_s1 = a->strides[1];
    const int b_s0 = b->strides[0];
    const int b_s1 = b->strides[1];
    const int out_s0 = out->strides[0];
    const int out_s1 = out->strides[1];

    /* dA = dOut @ B^T */
    if (a->requires_grad) {
        if (!a->grad) return; /* ensure_grad should have allocated */
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float acc = 0.0f;
                for (int j = 0; j < N; j++) {
                    int off_out = i * out_s0 + j * out_s1;
                    int off_b = k * b_s0 + j * b_s1;
                    acc += out->grad->data[off_out] * b->data[off_b];
                }
                int off_a = i * a_s0 + k * a_s1;
                a->grad->data[off_a] += acc;
            }
        }
    }

    /* dB = A^T @ dOut */
    if (b->requires_grad) {
        if (!b->grad) return;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int i = 0; i < M; i++) {
                    int off_a = i * a_s0 + k * a_s1;
                    int off_out = i * out_s0 + j * out_s1;
                    acc += a->data[off_a] * out->grad->data[off_out];
                }
                int off_b = k * b_s0 + j * b_s1;
                b->grad->data[off_b] += acc;
            }
        }
    }
}

/* CPU implementation kept as a static helper; public `tensor_matmul` will
 * dispatch to a backend implementation when appropriate. */
static Tensor *tensor_matmul_cpu(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int out_shape[2];
    out_shape[0] = a->shape[0];
    out_shape[1] = b->shape[1];

    Tensor *out = tensor_new(2, out_shape);
    if (!out) return NULL;
    out->requires_grad = (a->requires_grad || b->requires_grad);
    out->device = a->device;

    const int M = a->shape[0];
    const int K = a->shape[1];
    const int N = b->shape[1];

    const int a_s0 = a->strides[0];
    const int a_s1 = a->strides[1];
    const int b_s0 = b->strides[0];
    const int b_s1 = b->strides[1];
    const int out_s0 = out->strides[0];
    const int out_s1 = out->strides[1];

#ifdef _OPENMP
    int threads = 1;
    const char *env = getenv("TINYFIN_THREADS");
    if (env) {
        int t = atoi(env);
        if (t > 1) threads = t;
    }
    if (threads > 1) {
#pragma omp parallel for collapse(2) num_threads(threads)
    int tmp_fallback = 0;
    float *tmp = (float *)scratch_alloc((size_t)K * sizeof(float));
    if (!tmp) {
        tmp = (float *)malloc((size_t)K * sizeof(float));
        tmp_fallback = 1;
    }
    for (int i = 0; i < M; i++) {
        /* prefetch row of A */
        for (int k = 0; k < K; k++) {
            int off_a = i * a_s0 + k * a_s1;
            tmp[k] = a->data[off_a];
        }
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int off_b = k * b_s0 + j * b_s1;
                sum += tmp[k] * b->data[off_b];
            }
            int off_out = i * out_s0 + j * out_s1;
            out->data[off_out] = sum;
        }
    }
    /* scratch arena is reset elsewhere; free only if malloc fallback */
    if (tmp_fallback && tmp) {
        free(tmp);
    }
    scratch_reset(); /* release arena for subsequent ops */
    } else
#endif
    {
        float *tmp = (float *)scratch_alloc((size_t)K * sizeof(float));
        if (!tmp) {
            for (int i = 0; i < M; i++) {
                profiler_begin_op("matmul_inner");
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        int off_a = i * a_s0 + k * a_s1;
                        int off_b = k * b_s0 + j * b_s1;
                        sum += a->data[off_a] * b->data[off_b];
                    }
                    int off_out = i * out_s0 + j * out_s1;
                    out->data[off_out] = sum;
                }
                profiler_end_op();
            }
        } else {
            for (int i = 0; i < M; i++) {
                profiler_begin_op("matmul_inner");
                for (int k = 0; k < K; k++) {
                    int off_a = i * a_s0 + k * a_s1;
                    tmp[k] = a->data[off_a];
                }
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        int off_b = k * b_s0 + j * b_s1;
                        sum += tmp[k] * b->data[off_b];
                    }
                    int off_out = i * out_s0 + j * out_s1;
                    out->data[off_out] = sum;
                }
                profiler_end_op();
            }
        }
    }

    /* ensure scratch buffer is reusable even on serial path */
    scratch_reset();

    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (!n) return out;
        memset(n, 0, sizeof(*n));
        n->a = a;
        n->b = b;
        n->out = out;
        n->backward = matmul_bwd;
        n->n_inputs = 2;
        n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 2);
        n->inputs[0] = a;
        n->inputs[1] = b;
        n->visited = 0;
        n->hook = NULL;
        Tensor_attach_gradients(out, n);
    }

    return out;
}


Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    /* If both tensors are on GPU and a backend provides matmul, dispatch. */
    Backend *bk = NULL;
    int dev_a = a ? a->device : 0;
    int dev_b = b ? b->device : 0;
    if (dev_a == DEVICE_GPU && dev_b == DEVICE_GPU) {
        bk = backend_get();
        if (bk && bk->matmul) {
            Tensor *out = bk->matmul(a, b);
            if (out) return out;
        }
    }
    /* fallback to CPU implementation */
    return tensor_matmul_cpu(a, b);
}

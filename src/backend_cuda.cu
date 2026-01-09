#ifdef TINYFIN_ENABLE_CUDA
extern "C" {
#include "backend.h"
#include "tensor.h"
#include "autograd.h"
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* Simple, naive CUDA matmul kernel (C-order, stride-aware). */
static __global__ void matmul_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K,
                                     int a_s0, int a_s1,
                                     int b_s0, int b_s1,
                                     int c_s0, int c_s1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * a_s0 + k * a_s1] * B[k * b_s0 + col * b_s1];
    }
    C[row * c_s0 + col * c_s1] = acc;
}

static int cuda_ok(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[tinyfin][cuda] %s: %s\n", msg, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

/* Backward is kept on CPU for now; tensors live in host memory even when
 * device flag is DEVICE_GPU. */
static void matmul_bwd_cuda(AutogradNode *n) {
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

static Tensor *cuda_matmul(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "[tinyfin][cuda] matmul currently supports float32 only; falling back\n");
        return NULL;
    }

    const int M = a->shape[0];
    const int K = a->shape[1];
    const int N = b->shape[1];
    const int out_shape[2] = {M, N};

    Tensor *out = tensor_new(2, out_shape);
    if (!out) return NULL;
    out->requires_grad = (a->requires_grad || b->requires_grad);
    out->device = DEVICE_GPU;

    size_t bytes_a = (size_t)a->size * sizeof(float);
    size_t bytes_b = (size_t)b->size * sizeof(float);
    size_t bytes_c = (size_t)M * (size_t)N * sizeof(float);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float *dA = NULL, *dB = NULL, *dC = NULL;
    if (!cuda_ok(cudaMalloc((void **)&dA, bytes_a), "cudaMalloc A")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dB, bytes_b), "cudaMalloc B")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dC, bytes_c), "cudaMalloc C")) goto fail;

    if (!cuda_ok(cudaMemcpy(dA, a->data, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy A")) goto fail;
    if (!cuda_ok(cudaMemcpy(dB, b->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy B")) goto fail;

    matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K,
                                   a->strides[0], a->strides[1],
                                   b->strides[0], b->strides[1],
                                   N, 1);
    if (!cuda_ok(cudaGetLastError(), "matmul kernel launch")) goto fail;
    if (!cuda_ok(cudaMemcpy(out->data, dC, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy C->host")) goto fail;

    /* Keep device flag on output; gradients computed on host. */
    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (n) {
            memset(n, 0, sizeof(*n));
            n->a = a;
            n->b = b;
            n->out = out;
            n->backward = matmul_bwd_cuda;
            n->n_inputs = 2;
            n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 2);
            if (n->inputs) {
                n->inputs[0] = a;
                n->inputs[1] = b;
            }
            Tensor_attach_gradients(out, n);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return out;

fail:
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    tensor_free(out);
    return NULL;
}

/* ---------- simple elementwise add/mul (contiguous, float32, no broadcasting) ---------- */
static int is_contiguous(const Tensor *t) {
    if (!t) return 0;
    int *exp = (int *)malloc(sizeof(int) * t->ndim);
    if (!exp) return 0;
    tensor_contiguous_strides(t->shape, t->ndim, exp);
    int ok = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->strides[i] != exp[i]) { ok = 0; break; }
    }
    free(exp);
    return ok;
}

static __global__ void add_kernel(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

static __global__ void mul_kernel(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

static __global__ void mul_scalar_kernel(const float *a, float scalar, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * scalar;
}

static __global__ void add_bias_kernel(const float *a, const float *b, float *out,
                                       int total, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = a[idx] + b[idx % inner];
}

static __global__ void add_kernel_bcast(const float *a, const float *b, float *out,
                                        const int *shape, const int *stride_a, const int *stride_b,
                                        int ndim, int total, int is_mul) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int tmp = idx;
    int off_a = 0, off_b = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int coord = tmp % shape[d];
        tmp /= shape[d];
        off_a += coord * stride_a[d];
        off_b += coord * stride_b[d];
    }
    float va = a[off_a];
    float vb = b[off_b];
    out[idx] = is_mul ? (va * vb) : (va + vb);
}

static Tensor *cuda_elemwise(Tensor *a, Tensor *b, int is_mul) {
    if (!a || !b) return NULL;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "[tinyfin][cuda] add/mul requires float32 tensors; falling back\n");
        return NULL;
    }
    if (a->device != DEVICE_GPU || b->device != DEVICE_GPU) return NULL;

    int *out_shape = NULL;
    int out_ndim = 0;
    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return NULL;
    if (out_ndim <= 0 || out_ndim > 8) {
        fprintf(stderr, "[tinyfin][cuda] add/mul supports up to 8 dims; falling back\n");
        free(out_shape);
        return NULL;
    }

    Tensor *out = tensor_new(out_ndim, out_shape);
    if (!out) { free(out_shape); return NULL; }
    out->requires_grad = (a->requires_grad || b->requires_grad);
    out->dtype = static_cast<decltype(out->dtype)>(DTYPE_FLOAT32);
    out->device = DEVICE_GPU;

    /* fail-fast for non-contiguous tensors in broadcast path (stride kernel assumes contiguous storage) */
    int allow_bcast = is_contiguous(a) && is_contiguous(b);

    int same_shape = (a->ndim == b->ndim) ? 1 : 0;
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) { same_shape = 0; break; }
        }
    }

    /* fast path: contiguous, same-shape elementwise */
    if (same_shape) {
        size_t total = (size_t)out->size;
        size_t bytes = total * sizeof(float);
        float *dA = NULL, *dB = NULL, *dOut = NULL;
        if (!cuda_ok(cudaMalloc((void **)&dA, bytes), "cudaMalloc add/mul A")) goto fail_fast;
        if (!cuda_ok(cudaMalloc((void **)&dB, bytes), "cudaMalloc add/mul B")) goto fail_fast;
        if (!cuda_ok(cudaMalloc((void **)&dOut, bytes), "cudaMalloc add/mul out")) goto fail_fast;
        if (!cuda_ok(cudaMemcpy(dA, a->data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy add/mul A")) goto fail_fast;
        if (!cuda_ok(cudaMemcpy(dB, b->data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy add/mul B")) goto fail_fast;

        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        if (is_mul) {
            mul_kernel<<<blocks, threads>>>(dA, dB, dOut, (int)total);
        } else {
            add_kernel<<<blocks, threads>>>(dA, dB, dOut, (int)total);
        }
        if (!cuda_ok(cudaGetLastError(), "add/mul kernel launch")) goto fail_fast;
        if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy add/mul out")) goto fail_fast;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dOut);
        free(out_shape);
        return out;

    fail_fast:
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dOut) cudaFree(dOut);
        free(out_shape);
        tensor_free(out);
        return NULL;
    }

    /* fast path: scalar mul (contiguous) */
    if (is_mul && is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
        const Tensor *base = NULL;
        float scalar = 0.0f;
        if (b->ndim == 1 && b->shape[0] == 1) { base = a; scalar = b->data[0]; }
        else if (a->ndim == 1 && a->shape[0] == 1) { base = b; scalar = a->data[0]; }

        if (base && base->size == out->size && total <= (size_t)INT_MAX) {
            size_t bytes_base = (size_t)base->size * sizeof(float);
            float *dA = NULL, *dOut = NULL;
            if (!cuda_ok(cudaMalloc((void **)&dA, bytes_base), "cudaMalloc mul/scalar A")) goto scalar_fallback;
            if (!cuda_ok(cudaMalloc((void **)&dOut, bytes_out), "cudaMalloc mul/scalar out")) goto scalar_fallback;
            if (!cuda_ok(cudaMemcpy(dA, base->data, bytes_base, cudaMemcpyHostToDevice), "cudaMemcpy mul/scalar A")) goto scalar_fallback;

            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            mul_scalar_kernel<<<blocks, threads>>>(dA, scalar, dOut, (int)total);
            if (!cuda_ok(cudaGetLastError(), "mul/scalar kernel launch")) goto scalar_fallback;
            if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy mul/scalar out")) goto scalar_fallback;

            cudaFree(dA);
            cudaFree(dOut);
            free(out_shape);
            return out;

        scalar_fallback:
            if (dA) cudaFree(dA);
            if (dOut) cudaFree(dOut);
        }
    }

    /* fast path: bias add (contiguous, add-only, last-dim matches bias or scalar) */
    if (!is_mul && is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
        const Tensor *base = NULL;
        const Tensor *bias = NULL;
        int last = out_ndim - 1;
        if (b->ndim == 1 && (b->shape[0] == 1 || (out_ndim >= 1 && out_shape[last] == b->shape[0]))) {
            base = a;
            bias = b;
        } else if (a->ndim == 1 && (a->shape[0] == 1 || (out_ndim >= 1 && out_shape[last] == a->shape[0]))) {
            base = b;
            bias = a;
        }

        if (base && bias && base->size == out->size && bias->shape[0] > 0 && total <= (size_t)INT_MAX) {
            int inner = bias->shape[0];
            size_t bytes_base = (size_t)base->size * sizeof(float);
            size_t bytes_bias = (size_t)bias->size * sizeof(float);
            float *dA = NULL, *dB = NULL, *dOut = NULL;
            if (!cuda_ok(cudaMalloc((void **)&dA, bytes_base), "cudaMalloc add/bias A")) goto bias_fallback;
            if (!cuda_ok(cudaMalloc((void **)&dB, bytes_bias), "cudaMalloc add/bias B")) goto bias_fallback;
            if (!cuda_ok(cudaMalloc((void **)&dOut, bytes_out), "cudaMalloc add/bias out")) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(dA, base->data, bytes_base, cudaMemcpyHostToDevice), "cudaMemcpy add/bias A")) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(dB, bias->data, bytes_bias, cudaMemcpyHostToDevice), "cudaMemcpy add/bias B")) goto bias_fallback;

            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            add_bias_kernel<<<blocks, threads>>>(dA, dB, dOut, (int)total, inner);
            if (!cuda_ok(cudaGetLastError(), "add/bias kernel launch")) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy add/bias out")) goto bias_fallback;

            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dOut);
            free(out_shape);
            return out;

        bias_fallback:
            if (dA) cudaFree(dA);
            if (dB) cudaFree(dB);
            if (dOut) cudaFree(dOut);
        }
    }

    if (!allow_bcast) {
        fprintf(stderr, "[tinyfin][cuda] add/mul broadcast requires contiguous tensors; falling back\n");
        free(out_shape);
        tensor_free(out);
        return NULL;
    }

    /* build broadcast-aware strides */
    int stride_a[8];
    int stride_b[8];
    for (int i = 0; i < out_ndim; i++) {
        int idx_a = i - (out_ndim - a->ndim);
        int idx_b = i - (out_ndim - b->ndim);
        stride_a[i] = (idx_a < 0 || a->shape[idx_a] == 1) ? 0 : a->strides[idx_a];
        stride_b[i] = (idx_b < 0 || b->shape[idx_b] == 1) ? 0 : b->strides[idx_b];
    }

    size_t total = (size_t)out->size;
    if (total > (size_t)INT_MAX) {
        fprintf(stderr, "[tinyfin][cuda] add/mul size too large for CUDA kernel; falling back\n");
        free(out_shape);
        tensor_free(out);
        return NULL;
    }
    if (total == 0) {
        free(out_shape);
        return out;
    }
    size_t bytes_out = total * sizeof(float);
    size_t bytes_a = (size_t)a->size * sizeof(float);
    size_t bytes_b = (size_t)b->size * sizeof(float);
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    float *dA = NULL, *dB = NULL, *dOut = NULL;
    int *dShape = NULL, *dStrideA = NULL, *dStrideB = NULL;

    if (!cuda_ok(cudaMalloc((void **)&dA, bytes_a), "cudaMalloc add/mul A")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dB, bytes_b), "cudaMalloc add/mul B")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dOut, bytes_out), "cudaMalloc add/mul out")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dShape, sizeof(int) * (size_t)out_ndim), "cudaMalloc shape")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dStrideA, sizeof(int) * (size_t)out_ndim), "cudaMalloc strideA")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dStrideB, sizeof(int) * (size_t)out_ndim), "cudaMalloc strideB")) goto fail;

    if (!cuda_ok(cudaMemcpy(dA, a->data, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy add/mul A")) goto fail;
    if (!cuda_ok(cudaMemcpy(dB, b->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy add/mul B")) goto fail;
    if (!cuda_ok(cudaMemcpy(dShape, out_shape, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy shape")) goto fail;
    if (!cuda_ok(cudaMemcpy(dStrideA, stride_a, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy strideA")) goto fail;
    if (!cuda_ok(cudaMemcpy(dStrideB, stride_b, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy strideB")) goto fail;

    add_kernel_bcast<<<blocks, threads>>>(dA, dB, dOut, dShape, dStrideA, dStrideB, out_ndim, (int)total, is_mul);
    if (!cuda_ok(cudaGetLastError(), "add/mul kernel launch")) goto fail;

    if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy add/mul out")) goto fail;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    cudaFree(dShape);
    cudaFree(dStrideA);
    cudaFree(dStrideB);
    free(out_shape);
    return out;

fail:
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dOut) cudaFree(dOut);
    if (dShape) cudaFree(dShape);
    if (dStrideA) cudaFree(dStrideA);
    if (dStrideB) cudaFree(dStrideB);
    free(out_shape);
    tensor_free(out);
    return NULL;
}

static Tensor *cuda_add(Tensor *a, Tensor *b) { return cuda_elemwise(a, b, 0); }
static Tensor *cuda_mul(Tensor *a, Tensor *b) { return cuda_elemwise(a, b, 1); }

static __global__ void conv2d_kernel(const float *x, const float *w, const float *b, float *out,
                                     int N, int C_in, int H, int W,
                                     int C_out, int KH, int KW,
                                     int Hout, int Wout, int total) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;
    int tmp = idx;
    int wo = tmp % Wout; tmp /= Wout;
    int ho = tmp % Hout; tmp /= Hout;
    int oc = tmp % C_out; tmp /= C_out;
    int n = tmp;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int xi_h = ho + kh;
                int xi_w = wo + kw;
                size_t x_idx = ((size_t)n * C_in + ic) * H * W + xi_h * W + xi_w;
                size_t w_idx = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
                acc += x[x_idx] * w[w_idx];
            }
        }
    }
    if (b) acc += b[oc];
    size_t out_idx = ((size_t)n * C_out + oc) * Hout * Wout + ho * Wout + wo;
    out[out_idx] = acc;
}

/* Backward kept on CPU; input/output live in host memory even for GPU tensors. */
static void conv2d_bwd_cuda(AutogradNode *n) {
    if (!n) return;
    Tensor *x = n->a;
    Tensor *w = n->b;
    Tensor *b = n->bias;
    Tensor *y = n->out;
    if (!x || !w || !y || !y->grad) return;
    if (x->dtype != DTYPE_FLOAT32 || w->dtype != DTYPE_FLOAT32) return;

    int N = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int C_out = w->shape[0];
    int KH = w->shape[2];
    int KW = w->shape[3];
    int Hout = y->shape[2];
    int Wout = y->shape[3];

    if (w->requires_grad) {
        if (!w->grad) return;
        for (int n_i = 0; n_i < N; n_i++) {
            for (int oc = 0; oc < C_out; oc++) {
                for (int ic = 0; ic < C_in; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            size_t w_idx = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
                            float g = 0.0f;
                            for (int ho = 0; ho < Hout; ho++) {
                                for (int wo = 0; wo < Wout; wo++) {
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t out_idx = ((size_t)n_i * C_out + oc) * Hout * Wout + ho * Wout + wo;
                                    size_t x_idx = ((size_t)n_i * C_in + ic) * H * W + xi_h * W + xi_w;
                                    g += y->grad->data[out_idx] * x->data[x_idx];
                                }
                            }
                            w->grad->data[w_idx] += g;
                        }
                    }
                }
            }
        }
    }

    if (x->requires_grad) {
        if (!x->grad) return;
        for (int n_i = 0; n_i < N; n_i++) {
            for (int oc = 0; oc < C_out; oc++) {
                for (int ho = 0; ho < Hout; ho++) {
                    for (int wo = 0; wo < Wout; wo++) {
                        size_t out_idx = ((size_t)n_i * C_out + oc) * Hout * Wout + ho * Wout + wo;
                        float god = y->grad->data[out_idx];
                        for (int ic = 0; ic < C_in; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                for (int kw = 0; kw < KW; kw++) {
                                    int xi_h = ho + kh;
                                    int xi_w = wo + kw;
                                    size_t x_idx = ((size_t)n_i * C_in + ic) * H * W + xi_h * W + xi_w;
                                    size_t w_idx = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
                                    x->grad->data[x_idx] += god * w->data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (b && b->requires_grad) {
        if (!b->grad) return;
        for (int oc = 0; oc < C_out; oc++) {
            float gb = 0.0f;
            for (int n_i = 0; n_i < N; n_i++) {
                for (int ho = 0; ho < Hout; ho++) {
                    for (int wo = 0; wo < Wout; wo++) {
                        size_t out_idx = ((size_t)n_i * C_out + oc) * Hout * Wout + ho * Wout + wo;
                        gb += y->grad->data[out_idx];
                    }
                }
            }
            b->grad->data[oc] += gb;
        }
    }
}

static Tensor *cuda_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    if (!input || !weight) return NULL;
    if (input->dtype != DTYPE_FLOAT32 || weight->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "[tinyfin][cuda] conv2d supports float32 only; falling back\n");
        return NULL;
    }
    if (bias && bias->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "[tinyfin][cuda] conv2d bias must be float32; falling back\n");
        return NULL;
    }
    if (input->device != DEVICE_GPU || weight->device != DEVICE_GPU) return NULL;
    if (bias && bias->device != DEVICE_GPU) {
        fprintf(stderr, "[tinyfin][cuda] conv2d bias must be on GPU; falling back\n");
        return NULL;
    }
    if (input->ndim != 4 || weight->ndim != 4) {
        fprintf(stderr, "[tinyfin][cuda] conv2d expects 4D input/weight; falling back\n");
        return NULL;
    }
    if (!is_contiguous(input) || !is_contiguous(weight) || (bias && !is_contiguous(bias))) {
        fprintf(stderr, "[tinyfin][cuda] conv2d requires contiguous tensors; falling back\n");
        return NULL;
    }

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int C_out = weight->shape[0];
    if (weight->shape[1] != C_in) {
        fprintf(stderr, "[tinyfin][cuda] conv2d weight shape mismatch; falling back\n");
        return NULL;
    }
    if (bias && (bias->ndim != 1 || bias->shape[0] != C_out)) {
        fprintf(stderr, "[tinyfin][cuda] conv2d bias must be shape [C_out]; falling back\n");
        return NULL;
    }
    int KH = weight->shape[2];
    int KW = weight->shape[3];
    int Hout = H - KH + 1;
    int Wout = W - KW + 1;
    if (Hout <= 0 || Wout <= 0) return NULL;

    int out_shape[4] = {N, C_out, Hout, Wout};
    Tensor *out = tensor_new(4, out_shape);
    if (!out) return NULL;
    out->requires_grad = (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad));
    out->dtype = static_cast<decltype(out->dtype)>(DTYPE_FLOAT32);
    out->device = DEVICE_GPU;

    size_t bytes_x = (size_t)input->size * sizeof(float);
    size_t bytes_w = (size_t)weight->size * sizeof(float);
    size_t bytes_out = (size_t)out->size * sizeof(float);

    size_t total_sz = (size_t)N * (size_t)C_out * (size_t)Hout * (size_t)Wout;
    if (total_sz == 0 || total_sz > (size_t)INT_MAX) {
        fprintf(stderr, "[tinyfin][cuda] conv2d output too large; falling back\n");
        tensor_free(out);
        return NULL;
    }
    int total = (int)total_sz;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    float *dX = NULL, *dW = NULL, *dB = NULL, *dOut = NULL;
    if (!cuda_ok(cudaMalloc((void **)&dX, bytes_x), "cudaMalloc conv2d X")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dW, bytes_w), "cudaMalloc conv2d W")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dOut, bytes_out), "cudaMalloc conv2d out")) goto fail;
    if (!cuda_ok(cudaMemcpy(dX, input->data, bytes_x, cudaMemcpyHostToDevice), "cudaMemcpy conv2d X")) goto fail;
    if (!cuda_ok(cudaMemcpy(dW, weight->data, bytes_w, cudaMemcpyHostToDevice), "cudaMemcpy conv2d W")) goto fail;
    if (bias) {
        size_t bytes_b = (size_t)bias->size * sizeof(float);
        if (!cuda_ok(cudaMalloc((void **)&dB, bytes_b), "cudaMalloc conv2d bias")) goto fail;
        if (!cuda_ok(cudaMemcpy(dB, bias->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy conv2d bias")) goto fail;
    }

    /* naive direct conv: one thread per output element */
    conv2d_kernel<<<blocks, threads>>>(dX, dW, dB, dOut, N, C_in, H, W, C_out, KH, KW, Hout, Wout, total);
    if (!cuda_ok(cudaGetLastError(), "conv2d kernel launch")) goto fail;
    if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy conv2d out")) goto fail;

    cudaFree(dX); cudaFree(dW); cudaFree(dOut); if (dB) cudaFree(dB);

    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (n) {
            memset(n, 0, sizeof(*n));
            n->a = input;
            n->b = weight;
            n->bias = bias;
            n->out = out;
            n->backward = conv2d_bwd_cuda;
            n->n_inputs = 3;
            n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 3);
            if (n->inputs) {
                n->inputs[0] = input;
                n->inputs[1] = weight;
                n->inputs[2] = bias;
            }
            Tensor_attach_gradients(out, n);
        }
    }
    return out;

fail:
    if (dX) cudaFree(dX);
    if (dW) cudaFree(dW);
    if (dOut) cudaFree(dOut);
    if (dB) cudaFree(dB);
    tensor_free(out);
    return NULL;
}

static Backend cuda_backend = {
    .name = "cuda",
    .matmul = cuda_matmul,
    .conv2d = cuda_conv2d,
    .add = cuda_add,
    .mul = cuda_mul,
};

__attribute__((constructor))
static void register_cuda_backend(void) {
    backend_register(&cuda_backend);
}
#endif /* TINYFIN_ENABLE_CUDA */

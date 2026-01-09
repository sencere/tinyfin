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

typedef struct {
    void *ptr;
    size_t size;
    int in_use;
} CudaBuf;

static CudaBuf cuda_pool[16];

static int cuda_resident_enabled(void) {
    const char *env = getenv("TINYFIN_CUDA_RESIDENT");
    if (!env) return 0;
    return atoi(env) != 0;
}

static void cuda_pool_reset(void) {
    for (size_t i = 0; i < sizeof(cuda_pool) / sizeof(cuda_pool[0]); i++) {
        cuda_pool[i].in_use = 0;
    }
}

static void *cuda_buf_alloc(size_t bytes, int *transient) {
    if (!cuda_resident_enabled()) {
        void *ptr = NULL;
        if (!cuda_ok(cudaMalloc(&ptr, bytes), "cudaMalloc transient")) return NULL;
        if (transient) *transient = 1;
        return ptr;
    }

    for (size_t i = 0; i < sizeof(cuda_pool) / sizeof(cuda_pool[0]); i++) {
        if (!cuda_pool[i].in_use && cuda_pool[i].ptr && cuda_pool[i].size >= bytes) {
            cuda_pool[i].in_use = 1;
            if (transient) *transient = 0;
            return cuda_pool[i].ptr;
        }
    }
    for (size_t i = 0; i < sizeof(cuda_pool) / sizeof(cuda_pool[0]); i++) {
        if (!cuda_pool[i].in_use && !cuda_pool[i].ptr) {
            void *ptr = NULL;
            if (!cuda_ok(cudaMalloc(&ptr, bytes), "cudaMalloc resident")) return NULL;
            cuda_pool[i].ptr = ptr;
            cuda_pool[i].size = bytes;
            cuda_pool[i].in_use = 1;
            if (transient) *transient = 0;
            return ptr;
        }
    }

    void *ptr = NULL;
    if (!cuda_ok(cudaMalloc(&ptr, bytes), "cudaMalloc transient fallback")) return NULL;
    if (transient) *transient = 1;
    return ptr;
}

static void cuda_buf_free(void *ptr, int transient) {
    if (transient && ptr) cudaFree(ptr);
}

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
    int trans_a = 0, trans_b = 0, trans_c = 0;
    dA = (float *)cuda_buf_alloc(bytes_a, &trans_a);
    if (!dA) goto fail;
    dB = (float *)cuda_buf_alloc(bytes_b, &trans_b);
    if (!dB) goto fail;
    dC = (float *)cuda_buf_alloc(bytes_c, &trans_c);
    if (!dC) goto fail;

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

    cuda_buf_free(dA, trans_a);
    cuda_buf_free(dB, trans_b);
    cuda_buf_free(dC, trans_c);
    cuda_pool_reset();
    return out;

fail:
    if (dA) cuda_buf_free(dA, trans_a);
    if (dB) cuda_buf_free(dB, trans_b);
    if (dC) cuda_buf_free(dC, trans_c);
    cuda_pool_reset();
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
        int trans_a = 0, trans_b = 0, trans_out = 0;
        dA = (float *)cuda_buf_alloc(bytes, &trans_a);
        if (!dA) goto fail_fast;
        dB = (float *)cuda_buf_alloc(bytes, &trans_b);
        if (!dB) goto fail_fast;
        dOut = (float *)cuda_buf_alloc(bytes, &trans_out);
        if (!dOut) goto fail_fast;
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

        cuda_buf_free(dA, trans_a);
        cuda_buf_free(dB, trans_b);
        cuda_buf_free(dOut, trans_out);
        cuda_pool_reset();
        free(out_shape);
        return out;

    fail_fast:
        if (dA) cuda_buf_free(dA, trans_a);
        if (dB) cuda_buf_free(dB, trans_b);
        if (dOut) cuda_buf_free(dOut, trans_out);
        cuda_pool_reset();
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
            int trans_a = 0, trans_out = 0;
            dA = (float *)cuda_buf_alloc(bytes_base, &trans_a);
            if (!dA) goto scalar_fallback;
            dOut = (float *)cuda_buf_alloc(bytes_out, &trans_out);
            if (!dOut) goto scalar_fallback;
            if (!cuda_ok(cudaMemcpy(dA, base->data, bytes_base, cudaMemcpyHostToDevice), "cudaMemcpy mul/scalar A")) goto scalar_fallback;

            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            mul_scalar_kernel<<<blocks, threads>>>(dA, scalar, dOut, (int)total);
            if (!cuda_ok(cudaGetLastError(), "mul/scalar kernel launch")) goto scalar_fallback;
            if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy mul/scalar out")) goto scalar_fallback;

            cuda_buf_free(dA, trans_a);
            cuda_buf_free(dOut, trans_out);
            cuda_pool_reset();
            free(out_shape);
            return out;

        scalar_fallback:
            if (dA) cuda_buf_free(dA, trans_a);
            if (dOut) cuda_buf_free(dOut, trans_out);
            cuda_pool_reset();
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
            int trans_a = 0, trans_b = 0, trans_out = 0;
            dA = (float *)cuda_buf_alloc(bytes_base, &trans_a);
            if (!dA) goto bias_fallback;
            dB = (float *)cuda_buf_alloc(bytes_bias, &trans_b);
            if (!dB) goto bias_fallback;
            dOut = (float *)cuda_buf_alloc(bytes_out, &trans_out);
            if (!dOut) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(dA, base->data, bytes_base, cudaMemcpyHostToDevice), "cudaMemcpy add/bias A")) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(dB, bias->data, bytes_bias, cudaMemcpyHostToDevice), "cudaMemcpy add/bias B")) goto bias_fallback;

            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            add_bias_kernel<<<blocks, threads>>>(dA, dB, dOut, (int)total, inner);
            if (!cuda_ok(cudaGetLastError(), "add/bias kernel launch")) goto bias_fallback;
            if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy add/bias out")) goto bias_fallback;

            cuda_buf_free(dA, trans_a);
            cuda_buf_free(dB, trans_b);
            cuda_buf_free(dOut, trans_out);
            cuda_pool_reset();
            free(out_shape);
            return out;

        bias_fallback:
            if (dA) cuda_buf_free(dA, trans_a);
            if (dB) cuda_buf_free(dB, trans_b);
            if (dOut) cuda_buf_free(dOut, trans_out);
            cuda_pool_reset();
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
    int trans_a = 0, trans_b = 0, trans_out = 0, trans_shape = 0, trans_sa = 0, trans_sb = 0;

    dA = (float *)cuda_buf_alloc(bytes_a, &trans_a);
    if (!dA) goto fail;
    dB = (float *)cuda_buf_alloc(bytes_b, &trans_b);
    if (!dB) goto fail;
    dOut = (float *)cuda_buf_alloc(bytes_out, &trans_out);
    if (!dOut) goto fail;
    dShape = (int *)cuda_buf_alloc(sizeof(int) * (size_t)out_ndim, &trans_shape);
    if (!dShape) goto fail;
    dStrideA = (int *)cuda_buf_alloc(sizeof(int) * (size_t)out_ndim, &trans_sa);
    if (!dStrideA) goto fail;
    dStrideB = (int *)cuda_buf_alloc(sizeof(int) * (size_t)out_ndim, &trans_sb);
    if (!dStrideB) goto fail;

    if (!cuda_ok(cudaMemcpy(dA, a->data, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy add/mul A")) goto fail;
    if (!cuda_ok(cudaMemcpy(dB, b->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy add/mul B")) goto fail;
    if (!cuda_ok(cudaMemcpy(dShape, out_shape, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy shape")) goto fail;
    if (!cuda_ok(cudaMemcpy(dStrideA, stride_a, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy strideA")) goto fail;
    if (!cuda_ok(cudaMemcpy(dStrideB, stride_b, sizeof(int) * (size_t)out_ndim, cudaMemcpyHostToDevice), "cudaMemcpy strideB")) goto fail;

    add_kernel_bcast<<<blocks, threads>>>(dA, dB, dOut, dShape, dStrideA, dStrideB, out_ndim, (int)total, is_mul);
    if (!cuda_ok(cudaGetLastError(), "add/mul kernel launch")) goto fail;

    if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy add/mul out")) goto fail;

    cuda_buf_free(dA, trans_a);
    cuda_buf_free(dB, trans_b);
    cuda_buf_free(dOut, trans_out);
    cuda_buf_free(dShape, trans_shape);
    cuda_buf_free(dStrideA, trans_sa);
    cuda_buf_free(dStrideB, trans_sb);
    cuda_pool_reset();
    free(out_shape);
    return out;

fail:
    if (dA) cuda_buf_free(dA, trans_a);
    if (dB) cuda_buf_free(dB, trans_b);
    if (dOut) cuda_buf_free(dOut, trans_out);
    if (dShape) cuda_buf_free(dShape, trans_shape);
    if (dStrideA) cuda_buf_free(dStrideA, trans_sa);
    if (dStrideB) cuda_buf_free(dStrideB, trans_sb);
    cuda_pool_reset();
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

static __global__ void conv2d_bwd_w_kernel(const float *x, const float *grad_out, float *grad_w,
                                           int N, int C_in, int H, int W,
                                           int C_out, int KH, int KW,
                                           int Hout, int Wout, int total_w) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_w) return;
    int tmp = idx;
    int kw = tmp % KW; tmp /= KW;
    int kh = tmp % KH; tmp /= KH;
    int ic = tmp % C_in; tmp /= C_in;
    int oc = tmp;
    float acc = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                int xi_h = ho + kh;
                int xi_w = wo + kw;
                size_t x_idx = ((size_t)n * C_in + ic) * H * W + xi_h * W + xi_w;
                size_t out_idx = ((size_t)n * C_out + oc) * Hout * Wout + ho * Wout + wo;
                acc += grad_out[out_idx] * x[x_idx];
            }
        }
    }
    size_t w_idx = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
    grad_w[w_idx] = acc;
}

static __global__ void conv2d_bwd_x_kernel(const float *w, const float *grad_out, float *grad_x,
                                           int N, int C_in, int H, int W,
                                           int C_out, int KH, int KW,
                                           int Hout, int Wout, int total_x) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_x) return;
    int tmp = idx;
    int w_idx = tmp % W; tmp /= W;
    int h_idx = tmp % H; tmp /= H;
    int ic = tmp % C_in; tmp /= C_in;
    int n = tmp;
    float acc = 0.0f;
    for (int oc = 0; oc < C_out; oc++) {
        for (int kh = 0; kh < KH; kh++) {
            int ho = h_idx - kh;
            if (ho < 0 || ho >= Hout) continue;
            for (int kw = 0; kw < KW; kw++) {
                int wo = w_idx - kw;
                if (wo < 0 || wo >= Wout) continue;
                size_t out_idx = ((size_t)n * C_out + oc) * Hout * Wout + ho * Wout + wo;
                size_t w_off = ((size_t)oc * C_in + ic) * KH * KW + kh * KW + kw;
                acc += grad_out[out_idx] * w[w_off];
            }
        }
    }
    size_t x_off = ((size_t)n * C_in + ic) * H * W + h_idx * W + w_idx;
    grad_x[x_off] = acc;
}

static __global__ void conv2d_bwd_b_kernel(const float *grad_out, float *grad_b,
                                           int N, int C_out, int Hout, int Wout, int total_b) {
    int oc = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (oc >= total_b) return;
    float acc = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                size_t out_idx = ((size_t)n * C_out + oc) * Hout * Wout + ho * Wout + wo;
                acc += grad_out[out_idx];
            }
        }
    }
    grad_b[oc] = acc;
}

static void conv2d_bwd_cpu(AutogradNode *n) {
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

/* Backward uses CUDA kernels, but inputs/outputs live in host memory. */
static void conv2d_bwd_cuda(AutogradNode *n) {
    if (!n) return;
    Tensor *x = n->a;
    Tensor *w = n->b;
    Tensor *b = n->bias;
    Tensor *y = n->out;
    if (!x || !w || !y || !y->grad) return;
    if (x->dtype != DTYPE_FLOAT32 || w->dtype != DTYPE_FLOAT32) {
        conv2d_bwd_cpu(n);
        return;
    }

    int N = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int C_out = w->shape[0];
    int KH = w->shape[2];
    int KW = w->shape[3];
    int Hout = y->shape[2];
    int Wout = y->shape[3];

    int need_x = x->requires_grad && x->grad;
    int need_w = w->requires_grad && w->grad;
    int need_b = b && b->requires_grad && b->grad;
    if (!need_x && !need_w && !need_b) return;

    size_t bytes_x = (size_t)x->size * sizeof(float);
    size_t bytes_w = (size_t)w->size * sizeof(float);
    size_t bytes_out = (size_t)y->grad->size * sizeof(float);
    size_t bytes_b = (b ? (size_t)b->size * sizeof(float) : 0);

    float *dX = NULL, *dW = NULL, *dOut = NULL;
    float *dGradX = NULL, *dGradW = NULL, *dGradB = NULL;
    int trans_x = 0, trans_w = 0, trans_out = 0;
    int trans_gx = 0, trans_gw = 0, trans_gb = 0;
    int ok = 1;

    dX = (float *)cuda_buf_alloc(bytes_x, &trans_x);
    if (!dX) ok = 0;
    dW = (float *)cuda_buf_alloc(bytes_w, &trans_w);
    if (!dW) ok = 0;
    dOut = (float *)cuda_buf_alloc(bytes_out, &trans_out);
    if (!dOut) ok = 0;
    if (ok && !cuda_ok(cudaMemcpy(dX, x->data, bytes_x, cudaMemcpyHostToDevice), "cudaMemcpy conv2d bwd X")) ok = 0;
    if (ok && !cuda_ok(cudaMemcpy(dW, w->data, bytes_w, cudaMemcpyHostToDevice), "cudaMemcpy conv2d bwd W")) ok = 0;
    if (ok && !cuda_ok(cudaMemcpy(dOut, y->grad->data, bytes_out, cudaMemcpyHostToDevice), "cudaMemcpy conv2d bwd dOut")) ok = 0;

    if (ok && need_x) {
        dGradX = (float *)cuda_buf_alloc(bytes_x, &trans_gx);
        if (!dGradX) ok = 0;
        if (ok && !cuda_ok(cudaMemset(dGradX, 0, bytes_x), "cudaMemset conv2d bwd dX")) ok = 0;
    }
    if (ok && need_w) {
        dGradW = (float *)cuda_buf_alloc(bytes_w, &trans_gw);
        if (!dGradW) ok = 0;
        if (ok && !cuda_ok(cudaMemset(dGradW, 0, bytes_w), "cudaMemset conv2d bwd dW")) ok = 0;
    }
    if (ok && need_b) {
        dGradB = (float *)cuda_buf_alloc(bytes_b, &trans_gb);
        if (!dGradB) ok = 0;
        if (ok && !cuda_ok(cudaMemset(dGradB, 0, bytes_b), "cudaMemset conv2d bwd dB")) ok = 0;
    }

    int threads = 128;
    if (ok && need_w) {
        int total_w = C_out * C_in * KH * KW;
        int blocks = (total_w + threads - 1) / threads;
        conv2d_bwd_w_kernel<<<blocks, threads>>>(dX, dOut, dGradW, N, C_in, H, W,
                                                 C_out, KH, KW, Hout, Wout, total_w);
        if (!cuda_ok(cudaGetLastError(), "conv2d bwd dW kernel")) ok = 0;
    }

    if (ok && need_x) {
        int total_x = N * C_in * H * W;
        int blocks = (total_x + threads - 1) / threads;
        conv2d_bwd_x_kernel<<<blocks, threads>>>(dW, dOut, dGradX, N, C_in, H, W,
                                                 C_out, KH, KW, Hout, Wout, total_x);
        if (!cuda_ok(cudaGetLastError(), "conv2d bwd dX kernel")) ok = 0;
    }

    if (ok && need_b) {
        int total_b = C_out;
        int blocks = (total_b + threads - 1) / threads;
        conv2d_bwd_b_kernel<<<blocks, threads>>>(dOut, dGradB, N, C_out, Hout, Wout, total_b);
        if (!cuda_ok(cudaGetLastError(), "conv2d bwd dB kernel")) ok = 0;
    }

    float *tmp_w = NULL;
    float *tmp_x = NULL;
    float *tmp_b = NULL;

    if (ok && need_w) {
        tmp_w = (float *)malloc(bytes_w);
        if (!tmp_w) ok = 0;
        if (ok && !cuda_ok(cudaMemcpy(tmp_w, dGradW, bytes_w, cudaMemcpyDeviceToHost), "cudaMemcpy conv2d bwd dW->host")) ok = 0;
    }

    if (ok && need_x) {
        tmp_x = (float *)malloc(bytes_x);
        if (!tmp_x) ok = 0;
        if (ok && !cuda_ok(cudaMemcpy(tmp_x, dGradX, bytes_x, cudaMemcpyDeviceToHost), "cudaMemcpy conv2d bwd dX->host")) ok = 0;
    }

    if (ok && need_b) {
        tmp_b = (float *)malloc(bytes_b);
        if (!tmp_b) ok = 0;
        if (ok && !cuda_ok(cudaMemcpy(tmp_b, dGradB, bytes_b, cudaMemcpyDeviceToHost), "cudaMemcpy conv2d bwd dB->host")) ok = 0;
    }

    if (ok && need_w) {
        for (size_t i = 0; i < w->grad->size; i++) w->grad->data[i] += tmp_w[i];
    }
    if (ok && need_x) {
        for (size_t i = 0; i < x->grad->size; i++) x->grad->data[i] += tmp_x[i];
    }
    if (ok && need_b) {
        for (size_t i = 0; i < b->grad->size; i++) b->grad->data[i] += tmp_b[i];
    }

    free(tmp_w);
    free(tmp_x);
    free(tmp_b);

    if (dX) cuda_buf_free(dX, trans_x);
    if (dW) cuda_buf_free(dW, trans_w);
    if (dOut) cuda_buf_free(dOut, trans_out);
    if (dGradX) cuda_buf_free(dGradX, trans_gx);
    if (dGradW) cuda_buf_free(dGradW, trans_gw);
    if (dGradB) cuda_buf_free(dGradB, trans_gb);
    cuda_pool_reset();

    if (!ok) {
        fprintf(stderr, "[tinyfin][cuda] conv2d backward failed; falling back to CPU\n");
        conv2d_bwd_cpu(n);
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
    int trans_x = 0, trans_w = 0, trans_b = 0, trans_out = 0;
    dX = (float *)cuda_buf_alloc(bytes_x, &trans_x);
    if (!dX) goto fail;
    dW = (float *)cuda_buf_alloc(bytes_w, &trans_w);
    if (!dW) goto fail;
    dOut = (float *)cuda_buf_alloc(bytes_out, &trans_out);
    if (!dOut) goto fail;
    if (!cuda_ok(cudaMemcpy(dX, input->data, bytes_x, cudaMemcpyHostToDevice), "cudaMemcpy conv2d X")) goto fail;
    if (!cuda_ok(cudaMemcpy(dW, weight->data, bytes_w, cudaMemcpyHostToDevice), "cudaMemcpy conv2d W")) goto fail;
    if (bias) {
        size_t bytes_b = (size_t)bias->size * sizeof(float);
        dB = (float *)cuda_buf_alloc(bytes_b, &trans_b);
        if (!dB) goto fail;
        if (!cuda_ok(cudaMemcpy(dB, bias->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy conv2d bias")) goto fail;
    }

    /* naive direct conv: one thread per output element */
    conv2d_kernel<<<blocks, threads>>>(dX, dW, dB, dOut, N, C_in, H, W, C_out, KH, KW, Hout, Wout, total);
    if (!cuda_ok(cudaGetLastError(), "conv2d kernel launch")) goto fail;
    if (!cuda_ok(cudaMemcpy(out->data, dOut, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy conv2d out")) goto fail;

    cuda_buf_free(dX, trans_x);
    cuda_buf_free(dW, trans_w);
    cuda_buf_free(dOut, trans_out);
    if (dB) cuda_buf_free(dB, trans_b);
    cuda_pool_reset();

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
    if (dX) cuda_buf_free(dX, trans_x);
    if (dW) cuda_buf_free(dW, trans_w);
    if (dOut) cuda_buf_free(dOut, trans_out);
    if (dB) cuda_buf_free(dB, trans_b);
    cuda_pool_reset();
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

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

    float *dA = NULL, *dB = NULL, *dC = NULL;
    if (!cuda_ok(cudaMalloc((void **)&dA, bytes_a), "cudaMalloc A")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dB, bytes_b), "cudaMalloc B")) goto fail;
    if (!cuda_ok(cudaMalloc((void **)&dC, bytes_c), "cudaMalloc C")) goto fail;

    if (!cuda_ok(cudaMemcpy(dA, a->data, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy A")) goto fail;
    if (!cuda_ok(cudaMemcpy(dB, b->data, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy B")) goto fail;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
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

static Tensor *cuda_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    (void)input; (void)weight; (void)bias;
    /* Not implemented yet; fall back to CPU path. */
    return NULL;
}

static Backend cuda_backend = {
    .name = "cuda",
    .matmul = cuda_matmul,
    .conv2d = cuda_conv2d,
};

__attribute__((constructor))
static void register_cuda_backend(void) {
    backend_register(&cuda_backend);
}
#endif /* TINYFIN_ENABLE_CUDA */

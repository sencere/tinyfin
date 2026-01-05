#include "backend.h"
#include <stdio.h>

#ifdef TINYFIN_ENABLE_CUDA_STUB

static Tensor *cuda_stub_matmul(Tensor *a, Tensor *b) {
    /* TODO: replace with real CUDA kernel; return NULL to fall back to CPU. */
    fprintf(stderr, "[tinyfin] CUDA backend not available; falling back to CPU matmul\n");
    return NULL;
}

static Tensor *cuda_stub_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    fprintf(stderr, "[tinyfin] CUDA backend not available; falling back to CPU conv2d\n");
    return NULL;
}

static Backend cuda_backend = {
    .name = "cuda",
    .matmul = cuda_stub_matmul,
    .conv2d = cuda_stub_conv2d,
};

__attribute__((constructor))
static void register_cuda_backend(void) {
    backend_register(&cuda_backend);
}

#endif /* TINYFIN_ENABLE_CUDA_STUB */

#include "backend.h"
#include <stdio.h>

#ifdef TINYFIN_ENABLE_BLAS_STUB

static Tensor *blas_stub_matmul(Tensor *a, Tensor *b) {
    fprintf(stderr, "[tinyfin] BLAS backend not linked; falling back to CPU matmul\n");
    return NULL;
}

static Backend blas_backend = {
    .name = "blas",
    .matmul = blas_stub_matmul,
    .conv2d = NULL,
};

__attribute__((constructor))
static void register_blas_backend(void) {
    backend_register(&blas_backend);
}

#endif /* TINYFIN_ENABLE_BLAS_STUB */

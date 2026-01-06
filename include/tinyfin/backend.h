#ifndef TINYFIN_BACKEND_H
#define TINYFIN_BACKEND_H

#include "tensor.h"

typedef struct Backend {
    const char *name;
    /* optional device-specific matmul; if NULL CPU implementation is used */
    Tensor *(*matmul)(Tensor *a, Tensor *b);
    /* optional conv2d override */
    Tensor *(*conv2d)(Tensor *input, Tensor *weight, Tensor *bias);
    /* optional elementwise add/mul overrides (float32) */
    Tensor *(*add)(Tensor *a, Tensor *b);
    Tensor *(*mul)(Tensor *a, Tensor *b);
    /* future: memcpy, more elementwise kernels, etc. */
} Backend;

/* register / get current backend (default: NULL => CPU-only) */
void backend_set(Backend *b);
Backend *backend_get(void);
const char *backend_name(void);
int backend_register(Backend *b);
int backend_set_by_name(const char *name);

#endif

#include "backend.h"
#include <stdio.h>

#ifdef TINYFIN_ENABLE_OPENGL_STUB

static Tensor *gl_stub_matmul(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] OpenGL backend not available; falling back to CPU matmul\n");
    return NULL;
}

static Tensor *gl_stub_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    (void)input; (void)weight; (void)bias;
    fprintf(stderr, "[tinyfin] OpenGL backend not available; falling back to CPU conv2d\n");
    return NULL;
}

static Tensor *gl_stub_add(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] OpenGL backend not available; falling back to CPU add\n");
    return NULL;
}

static Tensor *gl_stub_mul(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] OpenGL backend not available; falling back to CPU mul\n");
    return NULL;
}

static Backend opengl_backend = {
    .name = "opengl",
    .matmul = gl_stub_matmul,
    .conv2d = gl_stub_conv2d,
    .add = gl_stub_add,
    .mul = gl_stub_mul,
};

__attribute__((constructor))
static void register_opengl_backend(void) {
    backend_register(&opengl_backend);
}

#endif /* TINYFIN_ENABLE_OPENGL_STUB */

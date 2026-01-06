#include "backend.h"
#include <stdio.h>

#ifdef TINYFIN_ENABLE_VULKAN_STUB

static Tensor *vk_stub_matmul(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] Vulkan backend not available; falling back to CPU matmul\n");
    return NULL;
}

static Tensor *vk_stub_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
    (void)input; (void)weight; (void)bias;
    fprintf(stderr, "[tinyfin] Vulkan backend not available; falling back to CPU conv2d\n");
    return NULL;
}

static Tensor *vk_stub_add(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] Vulkan backend not available; falling back to CPU add\n");
    return NULL;
}

static Tensor *vk_stub_mul(Tensor *a, Tensor *b) {
    (void)a; (void)b;
    fprintf(stderr, "[tinyfin] Vulkan backend not available; falling back to CPU mul\n");
    return NULL;
}

static Backend vulkan_backend = {
    .name = "vulkan",
    .matmul = vk_stub_matmul,
    .conv2d = vk_stub_conv2d,
    .add = vk_stub_add,
    .mul = vk_stub_mul,
};

__attribute__((constructor))
static void register_vulkan_backend(void) {
    backend_register(&vulkan_backend);
}

#endif /* TINYFIN_ENABLE_VULKAN_STUB */

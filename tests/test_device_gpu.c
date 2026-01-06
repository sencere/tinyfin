#include "tensor.h"
#include "ops_add.h"
#include <stdio.h>
#include <assert.h>

int main() {
    int shape[2] = {2, 2};
    Tensor *a = tensor_new(2, shape);
    Tensor *b = tensor_new(2, shape);
    for (size_t i = 0; i < a->size; i++) { a->data[i] = (float)(i+1); b->data[i] = (float)(2*(i+1)); }

    /* mark tensors as DEVICE_GPU (logical placeholder) */
    tensor_set_device(a, DEVICE_GPU);
    tensor_set_device(b, DEVICE_GPU);
    assert(tensor_get_device(a) == DEVICE_GPU);
    assert(tensor_get_device(b) == DEVICE_GPU);

    Tensor *c = tensor_add(a, b);
    if (!c) { printf("tensor_add failed on GPU-device tensors\n"); return 1; }

    /* output should inherit device */
    assert(tensor_get_device(c) == DEVICE_GPU);

    /* validate computation */
    for (size_t i = 0; i < c->size; i++) {
        float expected = a->data[i] + b->data[i];
        if (c->data[i] != expected) {
            printf("value mismatch at %zu: got=%f expected=%f\n", i, c->data[i], expected);
            return 2;
        }
    }

    tensor_free(a); tensor_free(b); tensor_free(c);
    printf("[test_device_gpu.c] PASS\n");
    return 0;
}

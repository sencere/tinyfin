#include "tensor.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

int main() {
    int shape[2] = {2, 3};
    Tensor *t = tensor_new(2, shape);
    assert(t->size == 6);
    tensor_fill(t, 1.5f);
    for (size_t i=0;i<t->size;i++)
        assert(t->data[i]==1.5f);
    tensor_print(t);
    tensor_free(t);
    printf("[test_tensor.c] PASS\n");
    return 0;
}

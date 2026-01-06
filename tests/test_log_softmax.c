#include "tensor.h"
#include "ops_softmax.h"
#include <stdio.h>
#include <math.h>

int main() {
    int shape[2] = {2, 3};
    Tensor *x = tensor_new(2, shape);
    float vals[6] = {0.1f, 2.0f, -1.0f, 0.3f, 0.2f, -0.5f};
    for (int i = 0; i < 6; i++) x->data[i] = vals[i];

    Tensor *p = tensor_softmax_autograd(x);
    Tensor *l = tensor_log_softmax(x);

    /* Check exp(log_softmax) equals softmax (approximately) */
    for (int i = 0; i < 6; i++) {
        float e = expf(l->data[i]);
        float s = p->data[i];
        if (fabsf(e - s) > 1e-6f) {
            printf("LOG_SOFTMAX exp mismatch: %f vs %f\n", e, s);
            return 1;
        }
    }

    tensor_free(p); tensor_free(l); tensor_free(x);
    printf("[test_log_softmax.c] PASS\n");
    return 0;
}

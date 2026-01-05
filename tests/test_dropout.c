#include "tensor.h"
#include "ops_dropout.h"
#include <stdio.h>
#include <math.h>

int main() {
    int shape[2] = {2,4};
    Tensor *x = tensor_new(2, shape);
    for (int i=0;i<8;i++) x->data[i] = 1.0f;

    Tensor *y = tensor_dropout(x, 0.5f, 1);
    if (!y) { printf("dropout returned NULL\n"); return 1; }
    int zeros = 0;
    for (int i=0;i<8;i++) if (y->data[i] == 0.0f) zeros++;
    if (zeros == 0) { printf("expected some zeros during training\n"); return 2; }

    tensor_free(y);
    Tensor *z = tensor_dropout(x, 0.5f, 0);
    for (int i=0;i<8;i++) if (fabsf(z->data[i] - x->data[i]) > 1e-6f) { printf("eval path differs\n"); return 3; }

    tensor_free(z); tensor_free(x);
    printf("[test_dropout.c] PASS\n");
    return 0;
}

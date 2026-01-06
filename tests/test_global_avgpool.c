#include "tensor.h"
#include "ops_avgpool.h"
#include <stdio.h>
#include <math.h>

int main() {
    int shape[4] = {2, 2, 2, 2};
    Tensor *x = tensor_new(4, shape);
    // Channel 0: all ones; Channel 1: increasing values
    float vals[16] = {
        1,1, 1,1,
        2,3, 4,5,

        1,1, 1,1,
        6,7, 8,9
    };
    for (int i = 0; i < 16; i++) x->data[i] = vals[i];

    Tensor *g = tensor_global_avgpool2d(x);
    if (!g) { printf("GLOB_AVGPOOL returned NULL\n"); return 1; }
    // g shape should be [2,2]
    float e00 = 1.0f; // average of ones
    float e01 = (2+3+4+5)/4.0f; // channel 1 first batch
    float e10 = 1.0f;
    float e11 = (6+7+8+9)/4.0f;

    if (fabsf(g->data[0] - e00) > 1e-6f) { printf("mismatch 0: %f vs %f\n", g->data[0], e00); return 2; }
    if (fabsf(g->data[1] - e01) > 1e-6f) { printf("mismatch 1: %f vs %f\n", g->data[1], e01); return 3; }
    if (fabsf(g->data[2] - e10) > 1e-6f) { printf("mismatch 2: %f vs %f\n", g->data[2], e10); return 4; }
    if (fabsf(g->data[3] - e11) > 1e-6f) { printf("mismatch 3: %f vs %f\n", g->data[3], e11); return 5; }

    tensor_free(g); tensor_free(x);
    printf("[test_global_avgpool.c] PASS\n");
    return 0;
}

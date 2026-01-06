#include "tensor.h"
#include "autograd.h"
#include "ops_mul.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS_NUM 1e-3f
#define TOL 1e-2f

static float numeric_second_grad_square(Tensor *a, size_t idx) {
    /* central difference on first-derivative of sum(a*a) w.r.t. a[idx] */
    float orig = a->data[idx];

    /* plus perturbation */
    a->data[idx] = orig + EPS_NUM;
    Tensor *out_p = tensor_mul(a, a);
    Tensor *s_p = tensor_sum(out_p);
    tensor_zero_grad(a); tensor_zero_grad(s_p);
    s_p->grad->data[0] = 1.0f;
    tensor_backward(s_p);
    float g_p = a->grad->data[idx];
    tensor_free(out_p); tensor_free(s_p);

    /* minus perturbation */
    a->data[idx] = orig - EPS_NUM;
    Tensor *out_m = tensor_mul(a, a);
    Tensor *s_m = tensor_sum(out_m);
    tensor_zero_grad(a); tensor_zero_grad(s_m);
    s_m->grad->data[0] = 1.0f;
    tensor_backward(s_m);
    float g_m = a->grad->data[idx];
    tensor_free(out_m); tensor_free(s_m);

    /* restore */
    a->data[idx] = orig;
    /* central difference of gradients approximates second derivative */
    return (g_p - g_m) / (2.0f * EPS_NUM);
}

int main() {
    int shape[2] = {2, 2};
    size_t size = 4;

    Tensor *a = tensor_new(2, shape);
    for (size_t i = 0; i < size; i++) a->data[i] = 0.5f * (i + 1);

    /* analytic first-order check (sanity) */
    a->requires_grad = 1;
    Tensor *out = tensor_mul(a, a);
    Tensor *loss = tensor_sum(out);
    tensor_zero_grad(a); tensor_zero_grad(loss);
    loss->grad->data[0] = 1.0f;
    tensor_backward(loss);
    for (size_t i = 0; i < size; i++) {
        float expected = 2.0f * a->data[i];
        if (fabsf(a->grad->data[i] - expected) > 1e-4f) {
            printf("FIRST-ORDER mismatch at %zu: got=%f expected=%f\n", i, a->grad->data[i], expected);
            return 1;
        }
    }

    /* numeric second-order check: d2/dx2 sum(x*x) == 2 */
    for (size_t i = 0; i < size; i++) {
        float s2 = numeric_second_grad_square(a, i);
        if (fabsf(s2 - 2.0f) > TOL) {
            printf("SECOND-ORDER mismatch at %zu: numeric=%f expected=2.0\n", i, s2);
            return 2;
        }
    }

    tensor_free(out);
    tensor_free(loss);
    tensor_free(a);

    printf("[test_second_order.c] PASS\n");
    return 0;
}

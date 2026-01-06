#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS_NUM 1e-3f
#define TOL 1e-3f

static float numeric_grad_add(Tensor *a, Tensor *b, size_t idx) {
    /* central difference on a[idx] for loss=sum(tensor_add(a,b)) */
    float orig = a->data[idx];
    a->data[idx] = orig + EPS_NUM;
    Tensor *out_p = tensor_add(a, b);
    Tensor *s_p = tensor_sum(out_p);
    float fp = s_p->data[0];
    tensor_free(out_p); tensor_free(s_p);

    a->data[idx] = orig - EPS_NUM;
    Tensor *out_m = tensor_add(a, b);
    Tensor *s_m = tensor_sum(out_m);
    float fm = s_m->data[0];
    tensor_free(out_m); tensor_free(s_m);

    a->data[idx] = orig;
    return (fp - fm) / (2.0f * EPS_NUM);
}

static float numeric_grad_mul(Tensor *a, Tensor *b, size_t idx) {
    float orig = a->data[idx];
    a->data[idx] = orig + EPS_NUM;
    Tensor *out_p = tensor_mul(a, b);
    Tensor *s_p = tensor_sum(out_p);
    float fp = s_p->data[0];
    tensor_free(out_p); tensor_free(s_p);

    a->data[idx] = orig - EPS_NUM;
    Tensor *out_m = tensor_mul(a, b);
    Tensor *s_m = tensor_sum(out_m);
    float fm = s_m->data[0];
    tensor_free(out_m); tensor_free(s_m);

    a->data[idx] = orig;
    return (fp - fm) / (2.0f * EPS_NUM);
}

int main() {
    int shape[2] = {2, 3};
    size_t size = 2*3;

    /* deterministic "random" values */
    Tensor *a = tensor_new(2, shape);
    Tensor *b = tensor_new(2, shape);
    for (size_t i = 0; i < size; i++) {
        a->data[i] = 0.1f*(i+1);
        b->data[i] = 0.2f*(i+2);
    }

    /* CHECK ADD */
    a->requires_grad = 1; b->requires_grad = 1;
    Tensor *out_add = tensor_add(a, b);
    Tensor *loss_add = tensor_sum(out_add);
    tensor_zero_grad(a); tensor_zero_grad(b); tensor_zero_grad(loss_add);
    /* set dL/dL = 1 for scalar loss */
    loss_add->grad->data[0] = 1.0f;
    tensor_backward(loss_add);

    for (size_t i = 0; i < size; i++) {
        float g_an = a->grad->data[i];
        float g_num = numeric_grad_add(a, b, i);
        if (fabsf(g_an - g_num) > TOL) {
            printf("ADD grad mismatch at %zu: analytic=%f numeric=%f\n", i, g_an, g_num);
            return 1;
        }
    }

    tensor_free(out_add);
    tensor_free(loss_add);

    /* CHECK MUL */
    /* reset grads and re-create graph to be safe */
    tensor_zero_grad(a); tensor_zero_grad(b);
    Tensor *out_mul = tensor_mul(a, b);
    Tensor *loss_mul = tensor_sum(out_mul);
    tensor_zero_grad(a); tensor_zero_grad(b); tensor_zero_grad(loss_mul);
    loss_mul->grad->data[0] = 1.0f;
    tensor_backward(loss_mul);

    for (size_t i = 0; i < size; i++) {
        float g_an = a->grad->data[i];
        float g_num = numeric_grad_mul(a, b, i);
        if (fabsf(g_an - g_num) > 1e-2f) {
            printf("MUL grad mismatch at %zu: analytic=%f numeric=%f\n", i, g_an, g_num);
            return 2;
        }
    }

    tensor_free(out_mul);
    tensor_free(loss_mul);
    tensor_free(a);
    tensor_free(b);

    printf("[test_numeric_gradcheck.c] PASS\n");
    return 0;
}

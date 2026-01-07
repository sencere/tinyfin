#include "tensor.h"
#include "autograd.h"
#include "ops_exp.h"
#include "ops_log.h"
#include "ops_reduce.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_exp_overflow() {
    int shape[2] = {2, 2};
    Tensor *a = tensor_new(2, shape);
    a->requires_grad = 1;
    a->data[0] = 100.0f;
    a->data[1] = 90.0f;
    a->data[2] = 0.0f;
    a->data[3] = -5.0f;

    Tensor *e = tensor_exp(a);
    float cap = expf(80.0f);
    assert(fabsf(e->data[0] - cap) < 1e-3f);
    assert(fabsf(e->data[1] - cap) < 1e-3f);
    assert(fabsf(e->data[2] - 1.0f) < 1e-6f);
    assert(fabsf(e->data[3] - expf(-5.0f)) < 1e-6f);

    Tensor *se = tensor_sum(e);
    tensor_zero_grad(a);
    tensor_backward(se);
    for (size_t i = 0; i < a->size; i++) {
        assert(isfinite(a->grad->data[i]));
    }

    tensor_free(a);
    tensor_free(e);
    tensor_free(se);
}

static void test_log_epsilon() {
    int shape[2] = {2, 2};
    Tensor *a = tensor_new(2, shape);
    a->requires_grad = 1;
    a->data[0] = 0.0f;
    a->data[1] = -1.0e-6f;
    a->data[2] = 1.0e-3f;
    a->data[3] = 1.0f;

    Tensor *l = tensor_log(a);
    float log_eps = logf(1.0e-12f);
    assert(fabsf(l->data[0] - log_eps) < 1e-5f);
    assert(fabsf(l->data[1] - log_eps) < 1e-5f);
    assert(fabsf(l->data[2] - logf(1.0e-3f)) < 1e-6f);
    assert(fabsf(l->data[3] - 0.0f) < 1e-6f);

    Tensor *sl = tensor_sum(l);
    tensor_zero_grad(a);
    tensor_backward(sl);
    assert(fabsf(a->grad->data[0]) < 1e-12f);
    assert(fabsf(a->grad->data[1]) < 1e-12f);
    assert(fabsf(a->grad->data[2] - 1000.0f) < 1e-2f);
    assert(fabsf(a->grad->data[3] - 1.0f) < 1e-6f);

    tensor_free(a);
    tensor_free(l);
    tensor_free(sl);
}

int main() {
    test_exp_overflow();
    test_log_epsilon();
    printf("[test_overflow.c] PASS\n");
    return 0;
}

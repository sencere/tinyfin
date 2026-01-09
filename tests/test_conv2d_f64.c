#include "tensor.h"
#include "autograd.h"
#include "ops_conv2d.h"
#include "ops_reduce.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void expect_close(double got, double exp, double tol) {
    assert(fabs(got - exp) <= tol);
}

int main() {
    int *in_shape = malloc(sizeof(int) * 4);
    in_shape[0] = 1; in_shape[1] = 1; in_shape[2] = 3; in_shape[3] = 3;
    Tensor *x = tensor_new(4, in_shape);
    tensor_set_dtype(x, DTYPE_FLOAT64);
    for (int i = 0; i < 9; i++) tensor_set_f64_at(x, (size_t)i, (double)(i + 1));

    int *w_shape = malloc(sizeof(int) * 4);
    w_shape[0] = 1; w_shape[1] = 1; w_shape[2] = 2; w_shape[3] = 2;
    Tensor *w = tensor_new(4, w_shape);
    tensor_set_dtype(w, DTYPE_FLOAT64);
    tensor_set_f64_at(w, 0, 1.0);
    tensor_set_f64_at(w, 1, 0.0);
    tensor_set_f64_at(w, 2, 0.0);
    tensor_set_f64_at(w, 3, 1.0);
    w->requires_grad = 1;

    int *b_shape = malloc(sizeof(int));
    b_shape[0] = 1;
    Tensor *b = tensor_new(1, b_shape);
    tensor_set_dtype(b, DTYPE_FLOAT64);
    tensor_set_f64_at(b, 0, 0.5);
    b->requires_grad = 1;

    Tensor *y = tensor_conv2d(x, w, b);
    expect_close(tensor_get_f64_at(y, 0), 6.5, 1e-9);
    expect_close(tensor_get_f64_at(y, 1), 8.5, 1e-9);
    expect_close(tensor_get_f64_at(y, 2), 12.5, 1e-9);
    expect_close(tensor_get_f64_at(y, 3), 14.5, 1e-9);

    Tensor *s = tensor_sum(y);
    tensor_zero_grad(w);
    tensor_zero_grad(b);
    tensor_backward(s);

    expect_close(tensor_get_f64_at(w->grad, 0), 12.0, 1e-9);
    expect_close(tensor_get_f64_at(w->grad, 1), 16.0, 1e-9);
    expect_close(tensor_get_f64_at(w->grad, 2), 24.0, 1e-9);
    expect_close(tensor_get_f64_at(w->grad, 3), 28.0, 1e-9);
    expect_close(tensor_get_f64_at(b->grad, 0), 4.0, 1e-9);

    tensor_free(x);
    tensor_free(w);
    tensor_free(b);
    tensor_free(y);
    tensor_free(s);
    free(in_shape);
    free(w_shape);
    free(b_shape);
    printf("[test_conv2d_f64.c] PASS\n");
    return 0;
}

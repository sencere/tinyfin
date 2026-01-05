#include "tensor.h"
#include "autograd.h"
#include "ops_abs.h"
#include "ops_neg.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *shape = (int *)malloc(sizeof(int)*2);
    shape[0]=1; shape[1]=2;

    /* Test abs */
    Tensor *a = tensor_new(2, shape);
    /* values: -2, 3 */
    a->data[0] = -2.0f; a->data[1] = 3.0f;
    a->requires_grad = 1;

    Tensor *abs_out = tensor_abs(a);
    /* forward check */
    assert(abs_out->data[0] == 2.0f);
    assert(abs_out->data[1] == 3.0f);

    Tensor *s = tensor_sum(abs_out);
    tensor_zero_grad(a);
    tensor_backward(s);

    /* gradients: sign(-2)=-1, sign(3)=1 */
    assert(a->grad->data[0] == -1.0f);
    assert(a->grad->data[1] == 1.0f);

    tensor_free(abs_out);
    tensor_free(s);
    tensor_free(a);

    /* Test neg */
    Tensor *b = tensor_new(2, shape);
    b->data[0] = 1.0f; b->data[1] = 2.0f;
    b->requires_grad = 1;

    Tensor *neg_out = tensor_neg(b);
    /* forward check */
    assert(neg_out->data[0] == -1.0f);
    assert(neg_out->data[1] == -2.0f);

    Tensor *s2 = tensor_sum(neg_out);
    tensor_zero_grad(b);
    tensor_backward(s2);

    /* d(sum(-b))/db = -1 each */
    assert(b->grad->data[0] == -1.0f);
    assert(b->grad->data[1] == -1.0f);

    tensor_free(neg_out);
    tensor_free(s2);
    tensor_free(b);
    free(shape);

    printf("[test_abs_neg.c] PASS\n");
    return 0;
}

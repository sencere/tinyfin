#include "tensor.h"
#include "autograd.h"
#include "ops_embedding.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *wshape = malloc(sizeof(int)*2);
    wshape[0]=2; wshape[1]=3; /* 2 embeddings, dim=3 */
    Tensor *W = tensor_new(2, wshape);
    /* set weights: row0 = [1,2,3], row1 = [4,5,6] */
    W->data[0]=1; W->data[1]=2; W->data[2]=3;
    W->data[3]=4; W->data[4]=5; W->data[5]=6;
    W->requires_grad = 1;

    int *ishape = malloc(sizeof(int)*1);
    ishape[0]=3; /* indices length 3 */
    Tensor *I = tensor_new(1, ishape);
    /* indices: [0,1,0] */
    I->data[0]=0.0f; I->data[1]=1.0f; I->data[2]=0.0f;

    Tensor *out = tensor_embedding(W, I);
    /* out should be shape [3,3] with rows [1,2,3], [4,5,6], [1,2,3] */
    assert(out->shape[out->ndim-1] == 3);
    /* check some values */
    assert(out->data[0] == 1.0f);
    assert(out->data[3] == 4.0f);
    assert(out->data[6] == 1.0f);

    Tensor *s = tensor_sum(out);
    tensor_zero_grad(W);
    tensor_backward(s);

    /* After summing all outputs, grad for W row0 should be count 2 per dim, row1 count 1 per dim */
    assert(W->grad->data[0] == 2.0f);
    assert(W->grad->data[1] == 2.0f);
    assert(W->grad->data[2] == 2.0f);
    assert(W->grad->data[3] == 1.0f);
    assert(W->grad->data[4] == 1.0f);
    assert(W->grad->data[5] == 1.0f);

    tensor_free(W); tensor_free(I); tensor_free(out); tensor_free(s); free(wshape); free(ishape);

    printf("[test_embedding.c] PASS\n");
    return 0;
}

#include "tensor.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int *shape = malloc(sizeof(int)*2);
    shape[0]=1; shape[1]=5;
    Tensor *t = tensor_new(2, shape);
    t->data[0]=1.0f; t->data[1]=3.0f; t->data[2]=2.0f; t->data[3]=5.0f; t->data[4]=0.5f;

    Tensor *amax = tensor_argmax(t);
    Tensor *amin = tensor_argmin(t);
    /* argmax should be index 3 (value 5.0), argmin index 4 (value 0.5) */
    assert((int)amax->data[0] == 3);
    assert((int)amin->data[0] == 4);

    tensor_free(t);
    tensor_free(amax);
    tensor_free(amin);
    free(shape);
    printf("[test_argmax.c] PASS\n");
    return 0;
}

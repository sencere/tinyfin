#include "tensor.h"
#include "autograd.h"
#include "ops_conv2d.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    /* Simple conv: batch=1, C_in=1, H=3,W=3; C_out=1, KH=2,KW=2 */
    int *in_shape = malloc(sizeof(int)*4);
    in_shape[0]=1; in_shape[1]=1; in_shape[2]=3; in_shape[3]=3;
    Tensor *x = tensor_new(4, in_shape);
    /* fill x with 1..9 */
    for (int i=0;i<9;i++) x->data[i] = (float)(i+1);

    int *w_shape = malloc(sizeof(int)*4);
    w_shape[0]=1; w_shape[1]=1; w_shape[2]=2; w_shape[3]=2;
    Tensor *w = tensor_new(4, w_shape);
    /* kernel [[1,0],[0,1]] */
    w->data[0]=1; w->data[1]=0; w->data[2]=0; w->data[3]=1;
    w->requires_grad = 1;

    int *b_shape = malloc(sizeof(int));
    b_shape[0] = 1;
    Tensor *b = tensor_new(1, b_shape);
    b->data[0] = 0.5f;
    b->requires_grad = 1;

    Tensor *y = tensor_conv2d(x, w, b);
    /* Expected output shape [1,1,2,2]; compute expected values manually and add bias */
    /* positions:
       top-left: 1*1 + 2*0 + 4*0 + 5*1 = 1+5=6
       top-right: 2*1 +3*0 +5*0 +6*1 =2+6=8
       bot-left:4*1+5*0+7*0+8*1=4+8=12
       bot-right:5*1+6*0+8*0+9*1=5+9=14 */
    assert(y->data[0]==6.5f);
    assert(y->data[1]==8.5f);
    assert(y->data[2]==12.5f);
    assert(y->data[3]==14.5f);

    Tensor *s = tensor_sum(y);
    tensor_zero_grad(w);
    tensor_zero_grad(b);
    tensor_backward(s);

    /* gradient for kernel should be sum over all patches of corresponding input entries
       i.e., for (0,0): positions of x contributing: 1,2,4,5 -> sum = 1+2+4+5 = 12
       but with kernel pattern, gradient for (0,0) accumulates overlay: careful compute expected
       Since kernel positions correspond to x positions: we can compute via manual sum: for (kh,kw)=(0,0)
       sum of x at positions [ (0,0),(0,1),(1,0),(1,1) ] = 1+2+4+5 = 12
       (0,1): positions (0,1),(0,2),(1,1),(1,2) = 2+3+5+6 = 16
       (1,0): (1,0),(1,1),(2,0),(2,1)=4+5+7+8=24
       (1,1): (1,1),(1,2),(2,1),(2,2)=5+6+8+9=28
    */
    assert(w->grad->data[0] == 12.0f);
    assert(w->grad->data[1] == 16.0f);
    assert(w->grad->data[2] == 24.0f);
    assert(w->grad->data[3] == 28.0f);

    /* bias grad should equal sum of output grads (4 elements) */
    assert(b->grad->data[0] == 4.0f);

    tensor_free(x); tensor_free(w); tensor_free(b); tensor_free(y); tensor_free(s);
    free(in_shape); free(w_shape); free(b_shape);
    printf("[test_conv2d.c] PASS\n");
    return 0;
}

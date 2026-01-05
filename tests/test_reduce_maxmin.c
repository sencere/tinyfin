#include "tensor.h"
#include "autograd.h"
#include "ops_reduce.h"
#include "ops_mul.h"
#include "ops_add.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-6

int main() {
    int shape[1] = {4};
    Tensor *t = tensor_new(1, shape);
    t->data[0] = 1.0f;
    t->data[1] = 3.0f;
    t->data[2] = 2.0f;
    t->data[3] = 3.0f; /* tie at indices 1 and 3 */

    t->requires_grad = 1;

    Tensor *mx = tensor_max(t);
    Tensor *mn = tensor_min(t);
    printf("created mx=%p mn=%p\n", (void*)mx, (void*)mn);

     /* prepare grads and run backward */
    if (!t->grad) t->grad = tensor_zeros(t->ndim, t->shape);
    printf("t->grad=%p\n", (void*)t->grad);
     tensor_zero_grad(t);

    if (!mx->grad) mx->grad = tensor_zeros(mx->ndim, mx->shape);
    if (!mn->grad) mn->grad = tensor_zeros(mn->ndim, mn->shape);
    printf("mx->grad=%p mn->grad=%p\n", (void*)mx->grad, (void*)mn->grad);
     tensor_zero_grad(mx);
     tensor_zero_grad(mn);

     mx->grad->data[0] = 1.0f;
     mn->grad->data[0] = 2.0f; /* arbitrary weight for min */

     /* mark outputs as requiring grad so tensor_backward will traverse
         (some ops don't set out->requires_grad consistently) */
     mx->requires_grad = 1;
    mx->requires_grad = 1;
    mn->requires_grad = 1;
    printf("mx.requires_grad=%d mn.requires_grad=%d\n", mx->requires_grad, mn->requires_grad);

    /* Build combined loss = mx + 2*mn so backward applies weights correctly */
    int sshape[1] = {1};
    Tensor *two = tensor_new(1, sshape);
    two->data[0] = 2.0f;
    Tensor *w = tensor_mul(mn, two);
    Tensor *loss = tensor_add(mx, w);
    if (!loss) { printf("failed to construct loss\n"); return 1; }

    /* debug: print grad_fn pointers */
    printf("mx.grad_fn=%p mn.grad_fn=%p t.grad_fn=%p\n", (void*)mx->grad_fn, (void*)mn->grad_fn, (void*)t->grad_fn);
    if (mx->grad_fn) printf("mx.n_inputs=%zu mx.inputs[0]=%p\n", mx->grad_fn->n_inputs, (void*)mx->grad_fn->inputs[0]);
    if (mn->grad_fn) printf("mn.n_inputs=%zu mn.inputs[0]=%p\n", mn->grad_fn->n_inputs, (void*)mn->grad_fn->inputs[0]);

    tensor_backward(loss);

     tensor_free(two);
     tensor_free(w);
     tensor_free(loss);

     printf("t grads: %.4f, %.4f, %.4f, %.4f\n",
              t->grad->data[0], t->grad->data[1], t->grad->data[2], t->grad->data[3]);

     /* According to first-index-wins policy, max grad goes to index 1 (first 3.0)
         min grad goes to index 0 (value 1.0) */
     assert(fabs(t->grad->data[0] - 2.0f) < EPS);
     assert(fabs(t->grad->data[1] - 1.0f) < EPS);
     assert(fabs(t->grad->data[2] - 0.0f) < EPS);
     assert(fabs(t->grad->data[3] - 0.0f) < EPS);

    tensor_free(t);
    tensor_free(mx);
    tensor_free(mn);

    printf("[test_reduce_maxmin.c] PASS\n");
    return 0;
}

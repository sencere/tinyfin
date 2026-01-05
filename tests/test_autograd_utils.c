#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>

int main() {
    printf("[test_autograd_utils.c] start\n");

    /* Test no_grad: disable autograd and ensure ops don't attach grad_fn */
    autograd_set_enabled(0);
    Tensor *a = tensor_new(1, (int[]){1}); a->data[0] = 2.0f; a->requires_grad = 1;
    Tensor *b = tensor_new(1, (int[]){1}); b->data[0] = 3.0f; b->requires_grad = 1;
    Tensor *c = tensor_add(a, b);
    /* when autograd disabled, attach should have skipped and out should not require grad */
    assert(c->grad_fn == NULL);
    assert(c->requires_grad == 0);

    tensor_free(a); tensor_free(b); tensor_free(c);

    /* Re-enable autograd and verify grad_fn is attached */
    autograd_set_enabled(1);
    a = tensor_new(1, (int[]){1}); a->data[0] = 2.0f; a->requires_grad = 1;
    b = tensor_new(1, (int[]){1}); b->data[0] = 3.0f; b->requires_grad = 1;
    c = tensor_add(a, b);
    assert(c->grad_fn != NULL);

    /* Test retain_graph: backward with retain=1 keeps nodes attached
       and retain=0 frees them. */
    Tensor *s = tensor_sum(c);
    tensor_zero_grad(c);
    tensor_zero_grad(a);
    tensor_zero_grad(b);

    tensor_backward_with_retain(s, 1);
    /* nodes should still be present */
    assert(c->grad_fn != NULL);

    /* now backward and free graph */
    tensor_backward_with_retain(s, 0);
    /* c's grad_fn should be cleared by cleanup */
    assert(c->grad_fn == NULL);

    tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(s);

    printf("[test_autograd_utils.c] PASS\n");
    return 0;
}

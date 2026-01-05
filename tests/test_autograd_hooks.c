#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_reduce.h"
#include <stdio.h>
#include <assert.h>

static int hook_counter = 0;

static void my_hook(AutogradNode *n) {
    (void)n;
    hook_counter++;
}

int main() {
    hook_counter = 0;
    Tensor *a = tensor_new(1, (int[]){1}); a->data[0] = 1.0f; a->requires_grad = 1;
    Tensor *b = tensor_new(1, (int[]){1}); b->data[0] = 2.0f; b->requires_grad = 1;
    Tensor *c = tensor_add(a, b);
    /* attach hook to c's autograd node */
    assert(c->grad_fn != NULL);
    c->grad_fn->hook = my_hook;

    Tensor *s = tensor_sum(c);
    tensor_zero_grad(a); tensor_zero_grad(b); tensor_zero_grad(c);
    tensor_backward_with_retain(s, 0);

    assert(hook_counter > 0);

    tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(s);
    printf("[test_autograd_hooks.c] PASS\n");
    return 0;
}

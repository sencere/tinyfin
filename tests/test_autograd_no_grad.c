#include "tensor.h"
#include "autograd.h"
#include <stdio.h>
#include <assert.h>

/* The py_api functions are included in the build and have C linkage. */
int py_autograd_push_disabled();
void py_autograd_pop(int prev);

int main() {
    printf("[test_autograd_no_grad.c] start\n");

    /* Ensure autograd is enabled initially */
    autograd_set_enabled(1);
    assert(autograd_get_enabled() == 1);

    int prev = py_autograd_push_disabled();
    assert(autograd_get_enabled() == 0);

    int prev2 = py_autograd_push_disabled();
    /* still disabled */
    assert(autograd_get_enabled() == 0);

    /* pop inner */
    py_autograd_pop(prev2);
    assert(autograd_get_enabled() == 0);

    /* pop outer restores previous state */
    py_autograd_pop(prev);
    assert(autograd_get_enabled() == 1);

    printf("[test_autograd_no_grad.c] PASS\n");
    return 0;
}

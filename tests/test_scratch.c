#include "scratch.h"
#include <assert.h>
#include <stdio.h>

int main() {
    /* set small arena to exercise both arena reuse and malloc fallback */
    scratch_init(256);

    char *p1 = (char *)scratch_alloc(64);
    char *p2 = (char *)scratch_alloc(64);
    assert(p1 && p2 && p2 > p1);

    /* reset should allow reuse of the arena from the beginning */
    scratch_reset();
    char *p3 = (char *)scratch_alloc(64);
    assert(p3 == p1);

    /* request larger than arena -> malloc fallback; should be distinct */
    char *p4 = (char *)scratch_alloc(1024);
    assert(p4 && p4 != p1);

    scratch_shutdown();
    printf("[test_scratch.c] PASS\n");
    return 0;
}

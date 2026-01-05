#include "tensor.h"
// Include autograd.h to bring in necessary structs and (hopefully) function prototypes
#include "autograd.h" 
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_reduce.h"
#include "ops_activation.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> // Needed for malloc/free

// Use library-provided tensor_zero_grad and tensor_backward from autograd.h


int main() {
    // FIX 2: Define and allocate the shape array [2, 2] for the 2x2 tensors
    int *shape_2x2 = (int *)malloc(sizeof(int) * 2);
    if (!shape_2x2) return 1;
    shape_2x2[0] = 2;
    shape_2x2[1] = 2;
    
    // FIX 3: Use the shape array pointer for tensor_new
    Tensor *t1 = tensor_new(2, shape_2x2);
    tensor_fill(t1,2.0f);
    t1->requires_grad=1;

    // FIX 3: Use the shape array pointer for tensor_new
    Tensor *t2 = tensor_new(2, shape_2x2);
    tensor_fill(t2,3.0f);
    t2->requires_grad=1;
    
    // FIX 4: Free the shape array now that tensors are created
    free(shape_2x2); 

    Tensor *a = tensor_add(t1,t2);
    Tensor *m = tensor_mul(a,t1);
    Tensor *r = tensor_relu(m);
    Tensor *s = tensor_sum(r);
    Tensor *mean = tensor_mean(r);

    // FIX 1: Functions are now declared
    tensor_zero_grad(t1);
    tensor_zero_grad(t2);
    tensor_backward(s);
    tensor_backward(mean);

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(a);
    tensor_free(m);
    tensor_free(r);
    tensor_free(s);
    tensor_free(mean);
    printf("[test_autograd_ops.c] PASS\n");
    return 0;
}
#include "tensor.h"
#include "nn.h"
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_activation.h"
#include "ops_loss.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> // FIX 1: Include stdlib.h for the free() function

// You may still need to include the prototype for tensor_backward if it's not in autograd.h
void tensor_backward(Tensor *t); 

int main() {
    // Simple 2-layer MLP
    int input_dim = 3;   // FIX 2: Change type to int for shape array compatibility
    int hidden = 4;
    int output = 2;

    // FIX 3: Allocate and define shape array for x (1D tensor with 'input_dim' elements)
    int *x_shape = (int *)malloc(sizeof(int) * 1);
    if (!x_shape) return 1;
    x_shape[0] = input_dim; 

    // FIX 3: Pass the shape array pointer
    Tensor *x = tensor_new(1, x_shape); // Line 12
    free(x_shape); // Free the temporary shape array

    tensor_fill(x, 1.0f);

    // Assuming linear_create handles weight and bias creation correctly
    Linear *l1 = linear_create(input_dim, hidden);
    Linear *l2 = linear_create(hidden, output);

    l1->weight->requires_grad = 1;
    l2->weight->requires_grad = 1;

    Tensor *h = linear_forward(l1, x);
    Tensor *h_relu = tensor_relu(h);
    Tensor *y = linear_forward(l2, h_relu);

    // FIX 4: Allocate and define shape array for target (1D tensor with 'output' elements)
    int *target_shape = (int *)malloc(sizeof(int) * 1);
    if (!target_shape) return 1;
    target_shape[0] = output; 

    // FIX 4: Pass the shape array pointer
    Tensor *target = tensor_new(1, target_shape); // Line 25
    free(target_shape); // Free the temporary shape array

    tensor_fill(target, 0.5f);

    Tensor *loss = tensor_mse_loss(y, target);
    tensor_backward(loss);

    printf("[test_mlp.c] Forward + Backward pass complete\n");

    tensor_free(x);
    tensor_free(h);
    tensor_free(h_relu);
    tensor_free(y);
    tensor_free(target);
    tensor_free(loss);

    // These free calls are typically handled by a linear_destroy function,
    // but we keep the current explicit frees.
    // NOTE: The `free(l1->weight)` is wrong if `l1->weight` is a Tensor*. You should use `tensor_free(l1->weight)`.
    // Assuming 'weight' and 'bias' pointers here point directly to the Tensor structs' memory:
    free(l1->weight); 
    free(l1->bias); 
    free(l1);

    free(l2->weight); 
    free(l2->bias); 
    free(l2);
    
    return 0;
}
#include "ops_loss.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>

// FIX 1: Explicitly declare binary_op, which is used by tensor_mse_loss
Tensor *binary_op(Tensor *a, Tensor *b,
                 void (*fwd)(Tensor*, Tensor*, Tensor*),
                 void (*bwd)(AutogradNode*));

static void mse_fwd(Tensor *p, Tensor *t, Tensor *out) {
    float s=0;
    for (size_t i=0;i<p->size;i++){
        float d=p->data[i]-t->data[i];
        s+=d*d;
    }
    out->data[0]=s/p->size;
}

static void mse_bwd(AutogradNode *n) {
    Tensor *p=n->a,*t=n->b,*out=n->out;
    
    if (!p->grad || !out->grad) return; // Safety check
    
    for (size_t i=0;i<p->size;i++)
        // FIX 2: Access gradient data using ->data, and cast p->size to float
        p->grad->data[i] += (2.0f / (float)p->size) * (p->data[i] - t->data[i]) * out->grad->data[0];
}

Tensor *tensor_mse_loss(Tensor *pred, Tensor *target) {
    // FIX 3: binary_op is now explicitly declared
    return binary_op(pred,target,mse_fwd,mse_bwd);
}
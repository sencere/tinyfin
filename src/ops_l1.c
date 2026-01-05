#include "ops_l1.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static void l1_fwd(Tensor *p, Tensor *t, Tensor *out, int reduction){
    float s=0.0f; size_t N=p->size;
    for (size_t i=0;i<N;i++) s += fabsf(p->data[i]-t->data[i]);
    if (reduction==0) out->data[0]=s; else out->data[0]=s/((reduction==1)?(float)N:(float)p->shape[0]);
}

static void l1_bwd(AutogradNode *n){
    Tensor *p=n->a,*t=n->b,*out=n->out; if (!out->grad) return;
    size_t N=p->size; int reduction=(int)(intptr_t)n->inputs[2]; float scale=(reduction==0)?1.0f:((reduction==1)?(1.0f/(float)N):(1.0f/(float)p->shape[0]));
    if (!p->grad) p->grad = tensor_zeros(p->ndim,p->shape);
    for (size_t i=0;i<N;i++){ float d = (p->data[i]>t->data[i])?1.0f:-1.0f; p->grad->data[i] += out->grad->data[0] * d * scale; }
}

Tensor *tensor_l1_loss(Tensor *pred, Tensor *target, int reduction){
    Tensor *out = tensor_new(1, (int[]){1}); out->requires_grad = (pred->requires_grad || target->requires_grad);
    l1_fwd(pred,target,out,reduction);
    if (out->requires_grad){ AutogradNode *n=malloc(sizeof(*n)); n->a=pred; n->b=target; n->out=out; n->backward=l1_bwd; n->n_inputs=2; n->inputs=malloc(sizeof(Tensor*)*3); n->inputs[0]=pred; n->inputs[1]=target; n->inputs[2]=(Tensor*)(intptr_t)reduction; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

#include "ops_huber.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <math.h>

static void huber_fwd(Tensor *p, Tensor *t, Tensor *out, float delta, int reduction){
    float s=0.0f; size_t N=p->size;
    for (size_t i=0;i<N;i++){ float d=fabsf(p->data[i]-t->data[i]); if (d<=delta) s+=0.5f*d*d; else s+=delta*(d-0.5f*delta); }
    if (reduction==0) out->data[0]=s; else out->data[0]=s/((reduction==1)?(float)N:(float)p->shape[0]);
}

static void huber_bwd(AutogradNode *n){
    Tensor *p=n->a,*t=n->b,*out=n->out; if (!out->grad) return;
    size_t N=p->size; int reduction=(int)(intptr_t)n->inputs[3]; float delta=* (float*)(n->inputs[2]); float scale=(reduction==0)?1.0f:((reduction==1)?(1.0f/(float)N):(1.0f/(float)p->shape[0]));
    if (!p->grad) p->grad = tensor_zeros(p->ndim,p->shape);
    for (size_t i=0;i<N;i++){ float diff = p->data[i]-t->data[i]; float d = fabsf(diff); float g = (d<=delta)?diff:delta*((diff>0)?1.0f:-1.0f); p->grad->data[i] += out->grad->data[0] * g * scale; }
}

Tensor *tensor_huber_loss(Tensor *pred, Tensor *target, float delta, int reduction){
    Tensor *out = tensor_new(1, (int[]){1}); out->requires_grad = (pred->requires_grad || target->requires_grad);
    huber_fwd(pred,target,out,delta,reduction);
    if (out->requires_grad){ AutogradNode *n=malloc(sizeof(*n)); n->a=pred; n->b=target; n->out=out; n->backward=huber_bwd; n->n_inputs=4; n->inputs=malloc(sizeof(Tensor*)*4); n->inputs[0]=pred; n->inputs[1]=target; n->inputs[2]=(Tensor*)malloc(sizeof(float)); *((float*)n->inputs[2])=delta; n->inputs[3]=(Tensor*)(intptr_t)reduction; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

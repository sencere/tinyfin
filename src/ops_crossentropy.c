#include "ops_crossentropy.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/* logits: [N,C], target: [N] with integer class indices */
static void ce_fwd(Tensor *logits, Tensor *target, Tensor *weight, Tensor *out, int reduction){
    int N = logits->shape[0];
    int C = logits->shape[1];
    float s = 0.0f;
    float wsum = 0.0f;
    for (int i=0;i<N;i++){
        size_t base = (size_t)i*C;
        /* stable log-sum-exp */
        float maxv = logits->data[base];
        for (int j=1;j<C;j++) if (logits->data[base+j]>maxv) maxv=logits->data[base+j];
        float sum = 0.0f;
        for (int j=0;j<C;j++) sum += expf(logits->data[base+j]-maxv);
        float logsum = maxv + logf(sum);
        int idx = (int)target->data[i];
        float l = logsum - logits->data[base + idx];
        float w = weight ? weight->data[idx] : 1.0f;
        float weighted = l * w;
        if (reduction==2) out->data[i] = weighted; else s += weighted;
        wsum += w;
    }
    if (reduction==0) out->data[0]=s;
    else if (reduction==1){
        float denom = weight ? (wsum>0.0f?wsum:(float)N) : (float)N;
        out->data[0]=s/denom;
    }
}

static void ce_bwd(AutogradNode *n){
    Tensor *logits = n->a; Tensor *target = n->b; Tensor *out=n->out;
    if (!out->grad) return;
    int N = logits->shape[0], C = logits->shape[1];
    if (!logits->grad) logits->grad = tensor_zeros(logits->ndim, logits->shape);

    Tensor *weight = n->inputs[2];
    int reduction = (int)(intptr_t)n->inputs[3];
    float denom = (float)N;
    if (reduction==1 && weight){
        float wsum = 0.0f;
        for (int i=0;i<N;i++){
            int idx = (int)target->data[i];
            wsum += weight->data[idx];
        }
        denom = (wsum>0.0f)?wsum:(float)N;
    }

    for (int i=0;i<N;i++){
        size_t base = (size_t)i*C;
        float maxv = logits->data[base];
        for (int j=1;j<C;j++) if (logits->data[base+j]>maxv) maxv=logits->data[base+j];
        float sum = 0.0f; for (int j=0;j<C;j++) sum+=expf(logits->data[base+j]-maxv);
        float upstream = (reduction==2)?out->grad->data[i]:out->grad->data[0];
        for (int j=0;j<C;j++) {
            float p = expf(logits->data[base+j]-maxv)/sum;
            int idx = (int)target->data[i];
            float g = p * out->grad->data[0];
            if (reduction==2) g = p * upstream;
            if (j==idx) g -= upstream;
            float w = weight ? weight->data[idx] : 1.0f;
            float scale = (reduction==0)?w:((reduction==1)?(w/denom):w);
            logits->grad->data[base+j] += g * scale;
        }
    }
}

Tensor *tensor_cross_entropy_logits(Tensor *logits, Tensor *target, Tensor *weight, int reduction){
    Tensor *out = (reduction==2) ? tensor_new(1, (int[]){logits->shape[0]}) : tensor_new(1, (int[]){1}); out->requires_grad = (logits->requires_grad || target->requires_grad);
    ce_fwd(logits,target,weight,out,reduction);
    if (out->requires_grad){ AutogradNode *n = malloc(sizeof(*n)); n->a=logits; n->b=target; n->out=out; n->backward=ce_bwd; n->n_inputs=3; n->inputs=malloc(sizeof(Tensor*)*4); n->inputs[0]=logits; n->inputs[1]=target; n->inputs[2]=weight; n->inputs[3]=(Tensor*)(intptr_t)reduction; n->visited=0; Tensor_attach_gradients(out,n); }
    return out;
}

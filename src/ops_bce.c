#include "ops_bce.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <math.h>

/* Numerically-stable BCE with logits helper */
static float sigmoid_f(float x){ return 1.0f/(1.0f+expf(-x)); }

static void bce_fwd(Tensor *pred, Tensor *target, Tensor *weight, Tensor *out, int logits, int reduction){
    float s = 0.0f;
    float wsum = 0.0f;
    size_t N = pred->size;
    for (size_t i=0;i<N;i++){
        float p = pred->data[i];
        float t = target->data[i];
        float loss;
        if (logits) {
            float maxv = fmaxf(0.0f, p);
            loss = maxv - p * t + logf(expf(-maxv) + expf(p - maxv));
        } else {
            p = fminf(fmaxf(p, 1e-7f), 1.0f-1e-7f);
            loss = - (t * logf(p) + (1.0f - t) * logf(1.0f - p));
        }
        float w = weight ? weight->data[i] : 1.0f;
        float weighted = loss * w;
        if (reduction == 2) out->data[i] = weighted;
        else s += weighted;
        wsum += w;
    }
    if (reduction == 0) out->data[0] = s;
    else if (reduction == 1) {
        float denom = weight ? (wsum > 0.0f ? wsum : (float)N) : (float)N;
        out->data[0] = s / denom;
    }
}

static void bce_bwd(AutogradNode *n){
    Tensor *pred = n->a, *target = n->b, *out = n->out;
    if (!out->grad) return;
    size_t N = pred->size;
    Tensor *weight = n->inputs[2];
    int logits = (int)(intptr_t)n->inputs[3];
    int reduction = (int)(intptr_t)n->inputs[4];

    if (!pred->grad) pred->grad = tensor_zeros(pred->ndim, pred->shape);

    float denom = (float)N;
    if (reduction == 1 && weight){
        float wsum = 0.0f;
        for (size_t i=0;i<N;i++) wsum += weight->data[i];
        denom = (wsum > 0.0f) ? wsum : (float)N;
    }

    for (size_t i=0;i<N;i++){
        float p = pred->data[i];
        float t = target->data[i];
        float dp;
        if (logits) {
            float s = sigmoid_f(p);
            dp = (s - t);
        } else {
            p = fminf(fmaxf(p, 1e-7f), 1.0f-1e-7f);
            dp = (p - t) / (p * (1.0f - p));
        }
        float upstream = (reduction==2) ? out->grad->data[i] : out->grad->data[0];
        float w = weight ? weight->data[i] : 1.0f;
        float scale = 1.0f;
        if (reduction==0) scale = w;
        else if (reduction==1) scale = w / denom;
        else if (reduction==2) scale = w;
        pred->grad->data[i] += upstream * dp * scale;
    }
}

Tensor *tensor_bce_loss(Tensor *pred, Tensor *target, Tensor *weight, int logits, int reduction){
    Tensor *out = (reduction==2) ? tensor_new(pred->ndim, pred->shape) : tensor_new(1, (int[]){1});
    out->requires_grad = (pred->requires_grad || target->requires_grad);
    bce_fwd(pred,target,weight,out,logits,reduction);
    if (out->requires_grad){
        AutogradNode *n = malloc(sizeof(*n)); n->a=pred; n->b=target; n->out=out; n->backward=bce_bwd; n->n_inputs=3; n->inputs=malloc(sizeof(Tensor*)*5); n->inputs[0]=pred; n->inputs[1]=target; n->inputs[2]=weight; n->inputs[3]=(Tensor*)(intptr_t)logits; n->inputs[4]=(Tensor*)(intptr_t)reduction; n->visited=0; Tensor_attach_gradients(out,n);
    }
    return out;
}

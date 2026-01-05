#include "ops_logsumexp.h"
#include "tensor.h"
#include <stdlib.h>
#include <math.h>

static void logsumexp_fwd(Tensor *t, Tensor *out){
    /* stable log-sum-exp: max trick */
    float maxv = t->data[0];
    for (size_t i=1;i<t->size;i++) if (t->data[i] > maxv) maxv = t->data[i];
    float sum = 0.0f;
    for (size_t i=0;i<t->size;i++) sum += expf(t->data[i] - maxv);
    out->data[0] = maxv + logf(sum);
}

static void logsumexp_bwd(AutogradNode *n){
    Tensor *t = n->a;
    Tensor *out = n->out;
    if (!out->grad) return;
    if (!t->grad) t->grad = tensor_zeros(t->ndim, t->shape);

    /* recompute softmax weights for gradient */
    float maxv = t->data[0];
    for (size_t i=1;i<t->size;i++) if (t->data[i] > maxv) maxv = t->data[i];
    float sum = 0.0f;
    for (size_t i=0;i<t->size;i++) sum += expf(t->data[i] - maxv);
    float g_out = out->grad->data[0];
    for (size_t i=0;i<t->size;i++){
        float p = expf(t->data[i] - maxv) / sum;
        t->grad->data[i] += g_out * p;
    }
}

Tensor *tensor_logsumexp(Tensor *t){
    int shape[1] = {1};
    Tensor *out = tensor_new(1, shape);
    if (!out) return NULL;
    out->requires_grad = t->requires_grad;
    logsumexp_fwd(t, out);
    if (out->requires_grad){
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t; n->out = out; n->backward = logsumexp_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = t; n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

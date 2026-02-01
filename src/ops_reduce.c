#include "ops_reduce.h"
#include "tensor.h"
#include "autograd.h"
#include "scratch.h"
#include <stdlib.h>
#include <stdio.h>

// Note: sum_fwd now takes the second Tensor* argument required by the function pointer 
// signature, and we cast it to void to suppress the 'unused parameter' warning.

static void sum_fwd(Tensor *a, Tensor *ignored, Tensor *out) {
    (void)ignored; // FIX: Silence 'unused parameter' warning
    float s = 0;
    for (size_t i = 0; i < a->size; i++) s += a->data[i];
    out->data[0] = s;
}

static void sum_bwd(AutogradNode *n) {
    Tensor *in = n->a;
    Tensor *out = n->out;
    
    if (!in || !out) return;
    if (in->requires_grad && in->grad && out->grad) {
        for (size_t i = 0; i < in->size; i++) 
            in->grad->data[i] += out->grad->data[0];
    }
}

Tensor *tensor_sum(Tensor *t) {
    // FIX: tensor_new expects a pointer to the shape array, not an integer.
    // Create a shape array for a scalar output [1].
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;

    Tensor *out = tensor_new(1, shape);
    if (!out) {
        scratch_reset();
        return NULL;
    }
    out->requires_grad = t->requires_grad;
    scratch_reset();
    
    sum_fwd(t, NULL, out);
    
    if (t->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t; n->out = out; n->backward = sum_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = (Tensor*)t;
        n->visited = 0;
        Tensor_attach_gradients(out,n);
    }
    return out;
}

static void sum_dim_bwd(AutogradNode *n) {
    Tensor *in = n->a;
    Tensor *out = n->out;
    if (!in || !out || !out->grad) return;
    int dim = (int)(intptr_t)n->inputs[1];
    if (!in->grad) in->grad = tensor_zeros(in->ndim, in->shape);
    if (!in->grad) return;

    if (in->ndim != 2 || out->ndim != 2) return;
    int N = in->shape[0];
    int M = in->shape[1];
    if (dim == 1) {
        for (int i = 0; i < N; i++) {
            float g = out->grad->data[i * out->shape[1]];
            for (int j = 0; j < M; j++) {
                in->grad->data[i * M + j] += g;
            }
        }
    } else {
        for (int j = 0; j < M; j++) {
            float g = out->grad->data[j];
            for (int i = 0; i < N; i++) {
                in->grad->data[i * M + j] += g;
            }
        }
    }
}

Tensor *tensor_sum_dim(Tensor *t, int dim) {
    if (!t) return NULL;
    if (t->ndim != 2) return NULL;
    if (dim != 0 && dim != 1) return NULL;

    int out_shape[2] = {t->shape[0], t->shape[1]};
    if (dim == 1) out_shape[1] = 1;
    else out_shape[0] = 1;

    Tensor *out = tensor_new(2, out_shape);
    if (!out) return NULL;
    out->requires_grad = t->requires_grad;

    int N = t->shape[0];
    int M = t->shape[1];
    if (dim == 1) {
        for (int i = 0; i < N; i++) {
            float s = 0.0f;
            for (int j = 0; j < M; j++) s += t->data[i * M + j];
            out->data[i * out->shape[1]] = s;
        }
    } else {
        for (int j = 0; j < M; j++) {
            float s = 0.0f;
            for (int i = 0; i < N; i++) s += t->data[i * M + j];
            out->data[j] = s;
        }
    }

    if (t->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t;
        n->out = out;
        n->backward = sum_dim_bwd;
        n->n_inputs = 1;
        n->inputs = malloc(sizeof(Tensor *) * 2);
        if (n->inputs) {
            n->inputs[0] = t;
            n->inputs[1] = (Tensor *)(intptr_t)dim;
        }
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

// Note: mean_fwd now takes the second Tensor* argument.
static void mean_fwd(Tensor *a, Tensor *ignored, Tensor *out) {
    (void)ignored; // FIX: Silence 'unused parameter' warning
    float s = 0;
    for (size_t i = 0; i < a->size; i++) s += a->data[i];
    out->data[0] = s / a->size;
}

static void mean_bwd(AutogradNode *n) {
    Tensor *in = n->a;
    Tensor *out = n->out;
    
    if (!out->grad) return; // Safety check
    
    // FIX: Access the float data using ->data, and cast in->size to float for division.
    float g = out->grad->data[0] / (float)in->size;
    
    if (in->requires_grad && in->grad) {
        // FIX: Access the actual float data using ->data.
        for (size_t i = 0; i < in->size; i++) 
            in->grad->data[i] += g;
    }
}

Tensor *tensor_mean(Tensor *t) {
    // FIX: tensor_new expects a pointer to the shape array.
    // Create a shape array for a scalar output [1].
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;

    Tensor *out = tensor_new(1, shape);
    if (!out) {
        scratch_reset();
        return NULL;
    }
    scratch_reset();
    out->requires_grad = t->requires_grad;
    
    mean_fwd(t, NULL, out);
    
    if (t->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t; n->out = out; n->backward = mean_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = (Tensor*)t;
        n->visited = 0;
        Tensor_attach_gradients(out,n);
    }
    return out; // <-- THIS LINE WAS LIKELY MISSING OR MISPLACED
}

/* forward declarations for max/min backward handlers */
static void max_bwd(AutogradNode *n);
static void min_bwd(AutogradNode *n);

/* ---------------- max / min / argmax / argmin ---------------- */

Tensor *tensor_max(Tensor *t) {
    if (!t) return NULL;
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;
    Tensor *out = tensor_new(1, shape);
    scratch_reset();
    if (!out) return NULL;

    float m = t->data[0];
    for (size_t i = 1; i < t->size; i++) if (t->data[i] > m) m = t->data[i];
    out->data[0] = m;

    if (t->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t; n->out = out; n->backward = max_bwd;
        n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = t; n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

static void max_bwd(AutogradNode *n) {
    if (!n || !n->a || !n->out || !n->out->grad) return;
    Tensor *in = n->a;

    /* find first index of max */
    size_t idx = 0;
    float m = in->data[0];
    for (size_t i = 1; i < in->size; i++) {
        if (in->data[i] > m) { m = in->data[i]; idx = i; }
    }

    float g = n->out->grad->data[0];
    if (!in->grad) in->grad = tensor_zeros(in->ndim, in->shape);
    if (!in->grad) return;
    in->grad->data[idx] += g;
}

Tensor *tensor_min(Tensor *t) {
    if (!t) return NULL;
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;
    Tensor *out = tensor_new(1, shape);
    scratch_reset();
    if (!out) return NULL;
    float m = t->data[0];
    for (size_t i = 1; i < t->size; i++) if (t->data[i] < m) m = t->data[i];
    out->data[0] = m;
    if (t->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->a = t; n->out = out; n->backward = min_bwd; n->n_inputs = 1; n->inputs = malloc(sizeof(Tensor*)); n->inputs[0] = t; n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    return out;
}

static void min_bwd(AutogradNode *n) {
    if (!n || !n->a || !n->out || !n->out->grad) return;
    Tensor *in = n->a;

    /* find first index of min */
    size_t idx = 0;
    float m = in->data[0];
    for (size_t i = 1; i < in->size; i++) {
        if (in->data[i] < m) { m = in->data[i]; idx = i; }
    }

    float g = n->out->grad->data[0];
    if (!in->grad) in->grad = tensor_zeros(in->ndim, in->shape);
    if (!in->grad) return;
    in->grad->data[idx] += g;
}

Tensor *tensor_argmax(Tensor *t) {
    if (!t) return NULL;
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;
    Tensor *out = tensor_new(1, shape);
    scratch_reset();
    if (!out) return NULL;
    size_t idx = 0; float m = t->data[0];
    for (size_t i = 1; i < t->size; i++) if (t->data[i] > m) { m = t->data[i]; idx = i; }
    out->data[0] = (float)idx; /* return index as float */
    return out; /* non-differentiable */
}

Tensor *tensor_argmin(Tensor *t) {
    if (!t) return NULL;
    int *shape = (int *)scratch_alloc(sizeof(int));
    if (!shape) return NULL;
    shape[0] = 1;
    Tensor *out = tensor_new(1, shape);
    scratch_reset();
    if (!out) return NULL;
    size_t idx = 0; float m = t->data[0];
    for (size_t i = 1; i < t->size; i++) if (t->data[i] < m) { m = t->data[i]; idx = i; }
    out->data[0] = (float)idx;
    return out;
}

#include "ops_embedding.h"
#include "tensor.h"
#include "autograd.h"
#include "scratch.h"
#include <stdlib.h>
#include <string.h>

static void embedding_bwd(AutogradNode *n) {
    if (!n || !n->out || !n->out->grad) return;
    Tensor *weights = n->a; /* weights */
    Tensor *indices = n->b; /* we store indices in n->b for backward */
    Tensor *out = n->out;
    if (!weights || !indices) return;

    /* ensure weights grad exists */
    if (!weights->grad) weights->grad = tensor_zeros(weights->ndim, weights->shape);
    if (!weights->grad) return;

    /* weights: [V, D], indices: arbitrary shape, out: indices.shape + [D] */
    int emb_dim = weights->shape[1];
    size_t n_indices = indices->size;

    /* for each position i in indices, get idx, then add out->grad segment to weights->grad[idx] */
    for (size_t i = 0; i < n_indices; i++) {
        /* read index value as int from indices->data (stored as float) */
        int idx = (int)tensor_get_f32_at(indices, i);
        if (idx < 0 || idx >= weights->shape[0]) continue;
        /* accumulate grad for dimension d */
        for (int d = 0; d < emb_dim; d++) {
            size_t out_offset = i * emb_dim + d;
            float g = tensor_get_f32_at(out->grad, out_offset);
            /* compute weight grad offset */
            size_t woff = (size_t)idx * emb_dim + d;
            float prev = tensor_get_f32_at(weights->grad, woff);
            tensor_set_f32_at(weights->grad, woff, prev + g);
        }
    }
}

Tensor *tensor_embedding(Tensor *weights, Tensor *indices) {
    if (!weights || !indices) return NULL;
    if (weights->ndim != 2) return NULL;

    int V = weights->shape[0];
    int D = weights->shape[1];

    /* build output shape = indices.shape + [D] */
    int out_ndim = indices->ndim + 1;
    int *out_shape = (int *)scratch_alloc(sizeof(int) * out_ndim);
    if (!out_shape) return NULL;
    for (int i = 0; i < indices->ndim; i++) out_shape[i] = indices->shape[i];
    out_shape[out_ndim - 1] = D;

    Tensor *out = tensor_new(out_ndim, out_shape);
    if (!out) {
        scratch_reset();
        return NULL;
    }
    out->requires_grad = weights->requires_grad;
    out->dtype = weights->dtype;
    tensor_set_dtype(out, weights->dtype);

    /* fill forward: for each linear position in indices, copy weights[idx, :]
       output is contiguous, so out->data layout is [n_indices, D] where n_indices = indices->size */
    size_t n_indices = indices->size;
    if (weights->dtype == DTYPE_FLOAT32) {
        float *w = (float *)weights->raw_data;
        float *o = (float *)out->raw_data;
        for (size_t i = 0; i < n_indices; i++) {
            int idx = (int)tensor_get_f32_at(indices, i);
            if (idx < 0 || idx >= V) {
                memset(o + i * (size_t)D, 0, sizeof(float) * (size_t)D);
                continue;
            }
            memcpy(o + i * (size_t)D, w + (size_t)idx * (size_t)D, sizeof(float) * (size_t)D);
        }
    } else {
        for (size_t i = 0; i < n_indices; i++) {
            int idx = (int)tensor_get_f32_at(indices, i);
            if (idx < 0 || idx >= V) {
                for (int d = 0; d < D; d++) tensor_set_f64_at(out, i * (size_t)D + d, 0.0);
                continue;
            }
            for (int d = 0; d < D; d++) {
                double v = tensor_get_f64_at(weights, (size_t)idx * (size_t)D + d);
                tensor_set_f64_at(out, i * (size_t)D + d, v);
            }
        }
    }

    scratch_reset();

    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        if (!n) return out;
        n->out = out;
        n->a = weights;
        n->b = indices; /* store indices for backward */
        n->backward = embedding_bwd;
        n->n_inputs = 2;
        n->inputs = malloc(sizeof(Tensor*) * 2);
        n->inputs[0] = weights;
        n->inputs[1] = indices;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }

    return out;
}

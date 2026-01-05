/* elementwise pow a^b (broadcasting). Note: b may be scalar or tensor. */
#include "ops_pow.h"
#include "tensor.h"
#include "autograd.h"
#include <stdlib.h>
#include <math.h>

static void unravel_index(int linear, const int *shape, int ndim, int *idx) {
    for (int d = ndim - 1; d >= 0; d--) {
        idx[d] = linear % shape[d];
        linear /= shape[d];
    }
}

static int tensor_offset(const Tensor *t, const int *idx) {
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    return off;
}

static Tensor *reduce_to_target(const Tensor *grad_out, const Tensor *target) {
    Tensor *g = tensor_zeros(target->ndim, target->shape);
    if (!g) return NULL;
    if (autograd_get_enabled()) {
        g->requires_grad = 1;
        g->grad = tensor_zeros(g->ndim, g->shape);
        if (g->grad) g->grad->requires_grad = 0;
    }
    int out_ndim = grad_out->ndim;
    int *idx_out = malloc(sizeof(int) * out_ndim);
    int *idx_t = malloc(sizeof(int) * target->ndim);
    if (!idx_out || !idx_t) {
        if (idx_out) free(idx_out);
        if (idx_t) free(idx_t);
        tensor_free(g);
        return NULL;
    }

    for (size_t i = 0; i < grad_out->size; i++) {
        unravel_index((int)i, grad_out->shape, out_ndim, idx_out);
        for (int d = 0; d < target->ndim; d++) {
            int out_d = d + (out_ndim - target->ndim);
            idx_t[d] = (out_d < 0 || target->shape[d] == 1) ? 0 : idx_out[out_d];
        }
        int off = tensor_offset(g, idx_t);
        g->data[off] += grad_out->data[i];
    }

    free(idx_out);
    free(idx_t);
    return g;
}

static void pow_fwd(Tensor *a, Tensor *b, Tensor *out) {
    int *out_shape = NULL;
    int out_ndim = 0;
    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return;

    Tensor ea = tensor_expand(a, out_shape, out->ndim);
    Tensor eb = tensor_expand(b, out_shape, out->ndim);

    int *idx = malloc(sizeof(int) * out->ndim);
    for (size_t i = 0; i < out->size; i++) {
        unravel_index((int)i, out_shape, out->ndim, idx);
        int ia = tensor_offset(&ea, idx);
        int ib = tensor_offset(&eb, idx);
        out->data[i] = powf(ea.data[ia], eb.data[ib]);
    }

    free(idx);
    free(out_shape);
}

static void pow_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    if (!out || !out->grad) return;

    int *out_shape = NULL;
    int out_ndim = 0;
    if (!tensor_broadcast_shape(n->a, n->b, &out_shape, &out_ndim)) return;

    Tensor ea = tensor_expand(n->a, out_shape, out->ndim);
    Tensor eb = tensor_expand(n->b, out_shape, out->ndim);

    int *idx = malloc(sizeof(int) * out->ndim);
    int _record = autograd_get_enabled();
    Tensor *gA_bcast = (n->a && n->a->requires_grad) ? tensor_new(out->ndim, out_shape) : NULL;
    if (gA_bcast && _record) {
        gA_bcast->requires_grad = 1;
        gA_bcast->grad = tensor_zeros(gA_bcast->ndim, gA_bcast->shape);
        if (gA_bcast->grad) gA_bcast->grad->requires_grad = 0;
    }
    Tensor *gB_bcast = (n->b && n->b->requires_grad) ? tensor_new(out->ndim, out_shape) : NULL;
    if (gB_bcast && _record) {
        gB_bcast->requires_grad = 1;
        gB_bcast->grad = tensor_zeros(gB_bcast->ndim, gB_bcast->shape);
        if (gB_bcast->grad) gB_bcast->grad->requires_grad = 0;
    }

    for (size_t i = 0; i < out->size; i++) {
        unravel_index((int)i, out_shape, out->ndim, idx);
        int ia = tensor_offset(&ea, idx);
        int ib = tensor_offset(&eb, idx);
        float a_val = ea.data[ia];
        float b_val = eb.data[ib];
        float y = out->data[i];
        float go = out->grad->data[i];
        if (gA_bcast) gA_bcast->data[i] = go * b_val * powf(a_val, b_val - 1.0f);
        if (gB_bcast) gB_bcast->data[i] = go * (isfinite(a_val) && a_val > 0 ? y * logf(a_val) : 0.0f);
    }

    if (n->a && n->a->requires_grad) {
        Tensor *ga = reduce_to_target(gA_bcast, n->a);
        if (ga) {
            if (!n->a->grad) n->a->grad = ga;
            else {
                for (size_t i = 0; i < n->a->size; i++) n->a->grad->data[i] += ga->data[i];
                tensor_free(ga);
            }
        }
    }

    if (n->b && n->b->requires_grad) {
        Tensor *gb = reduce_to_target(gB_bcast, n->b);
        if (gb) {
            if (!n->b->grad) n->b->grad = gb;
            else {
                for (size_t i = 0; i < n->b->size; i++) n->b->grad->data[i] += gb->data[i];
                tensor_free(gb);
            }
        }
    }

    if (gA_bcast) tensor_free(gA_bcast);
    if (gB_bcast) tensor_free(gB_bcast);
    free(idx);
    free(out_shape);
}

Tensor *tensor_pow(Tensor *a, Tensor *b) {
    int *out_shape = NULL;
    int out_ndim = 0;
    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return NULL;
    Tensor *out = tensor_new(out_ndim, out_shape);
    if (!out) { free(out_shape); return NULL; }
    out->requires_grad = (a->requires_grad || b->requires_grad);
    pow_fwd(a, b, out);
    if (out->requires_grad) {
        AutogradNode *n = malloc(sizeof(*n));
        n->out = out;
        n->a = a;
        n->b = b;
        n->backward = pow_bwd;
        n->n_inputs = 2;
        n->inputs = malloc(sizeof(Tensor *) * 2);
        n->inputs[0] = a;
        n->inputs[1] = b;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }
    free(out_shape);
    return out;
}

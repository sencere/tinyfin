#include "ops_add.h"
#include "tensor.h"
#include "autograd.h"
#include "graph.h"
#include "backend.h"
#include "scratch.h"
#include <stdlib.h>
#include <stdio.h>

/* ---------- small local indexing helpers ---------- */

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

/* Reduce a broadcasted grad (shape == out_shape) back to target's shape */
static Tensor *reduce_to_target(const Tensor *grad_out, const Tensor *target) {
    Tensor *g = tensor_zeros(target->ndim, target->shape);
    if (!g) return NULL;

    int out_ndim = grad_out->ndim;
    int *idx_out = (int *)scratch_alloc(sizeof(int) * out_ndim);
    if (!idx_out) { tensor_free(g); return NULL; }

    int *idx_t = (int *)scratch_alloc(sizeof(int) * target->ndim);
    if (!idx_t) {
        scratch_reset();
        tensor_free(g);
        return NULL;
    }

    for (size_t i = 0; i < grad_out->size; i++) {
        unravel_index((int)i, grad_out->shape, out_ndim, idx_out);

        /* build target index (align from right) */
        for (int d = 0; d < target->ndim; d++) {
            int out_d = d + (out_ndim - target->ndim);
            idx_t[d] = (out_d < 0 || target->shape[d] == 1) ? 0 : idx_out[out_d];
        }

        int off_t = tensor_offset(g, idx_t);
        g->data[off_t] += grad_out->data[i];
    }

    scratch_reset();
    return g;
}

/* -------------------- forward / backward for add -------------------- */

static void add_fwd(Tensor *a, Tensor *b, Tensor *out) {
    int *out_shape = NULL;
    int out_ndim = 0;

    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return;

    Tensor ea = tensor_expand(a, out_shape, out->ndim);
    Tensor eb = tensor_expand(b, out_shape, out->ndim);

    int *idx = (int *)scratch_alloc(sizeof(int) * out->ndim);
    if (!idx) { free(out_shape); return; }

    for (size_t i = 0; i < out->size; i++) {
        unravel_index((int)i, out_shape, out->ndim, idx);
        int ia = tensor_offset(&ea, idx);
        int ib = tensor_offset(&eb, idx);
        if (out->dtype == DTYPE_FLOAT32) {
            float va = tensor_get_f32_at(&ea, (size_t)ia);
            float vb = tensor_get_f32_at(&eb, (size_t)ib);
            tensor_set_f32_at(out, i, va + vb);
        } else {
            double va = tensor_get_f64_at(&ea, (size_t)ia);
            double vb = tensor_get_f64_at(&eb, (size_t)ib);
            tensor_set_f64_at(out, i, va + vb);
        }
    }

    scratch_reset();
    free(out_shape);
}

static void add_bwd(AutogradNode *n) {
    Tensor *out = n->out;
    if (!out || !out->grad) return;

    /* add_bwd called */

    if (n->a && n->a->requires_grad) {
        Tensor *ga = reduce_to_target(out->grad, n->a);
        if (ga) {
            if (!n->a->grad) n->a->grad = ga;
            else {
                for (size_t i = 0; i < n->a->size; i++) n->a->grad->data[i] += ga->data[i];
                tensor_free(ga);
            }
        }
    }

    if (n->b && n->b->requires_grad) {
        Tensor *gb = reduce_to_target(out->grad, n->b);
        if (gb) {
            if (!n->b->grad) n->b->grad = gb;
            else {
                for (size_t i = 0; i < n->b->size; i++) n->b->grad->data[i] += gb->data[i];
                tensor_free(gb);
            }
        }
    }
}

/* -------------------- public entry point -------------------- */

Tensor *tensor_add(Tensor *a, Tensor *b) {
    int *out_shape = NULL;
    int out_ndim = 0;

    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return NULL;

    /* simple device check: both inputs must be on same device for now */
    if (a->device != b->device) { free(out_shape); return NULL; }

    Tensor *out = NULL;

    /* Attempt backend for float32 tensors (broadcast handled in backend). */
    if (a->dtype == DTYPE_FLOAT32 && b->dtype == DTYPE_FLOAT32) {
        Backend *bk = backend_get();
        if (bk && bk->add) {
            out = bk->add(a, b);
            if (out) {
                out->requires_grad = (a->requires_grad || b->requires_grad);
            }
        }
    }

    if (!out) {
        out = tensor_new(out_ndim, out_shape);
        if (!out) { free(out_shape); return NULL; }
        out->requires_grad = (a->requires_grad || b->requires_grad);
        out->dtype = a->dtype;
        out->device = a->device;
        tensor_set_dtype(out, a->dtype);
        add_fwd(a, b, out);
    }

    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (!n) { free(out_shape); return out; }
        n->out = out;
        n->a = a;
        n->b = b;
        n->backward = add_bwd;
        n->n_inputs = 2;
        n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 2);
        n->inputs[0] = a;
        n->inputs[1] = b;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }

    {
        Tensor *inputs[2] = {a, b};
        graph_record_op(GRAPH_OP_ADD, out, inputs, 2);
    }

    free(out_shape);
    return out;
}

/* Generic binary op helper used by losses and other ops */
Tensor *binary_op(Tensor *a, Tensor *b,
                  void (*fwd)(Tensor*, Tensor*, Tensor*),
                  void (*bwd)(AutogradNode*)) {
    int *out_shape = NULL;
    int out_ndim = 0;

    if (!tensor_broadcast_shape(a, b, &out_shape, &out_ndim)) return NULL;

    Tensor *out = tensor_new(out_ndim, out_shape);
    if (!out) { free(out_shape); return NULL; }
    out->requires_grad = (a->requires_grad || b->requires_grad);

    fwd(a, b, out);

    if (out->requires_grad) {
        AutogradNode *n = (AutogradNode *)malloc(sizeof(*n));
        if (!n) { free(out_shape); return out; }
        n->out = out;
        n->a = a;
        n->b = b;
        n->backward = bwd;
        n->n_inputs = 2;
        n->inputs = (Tensor **)malloc(sizeof(Tensor *) * 2);
        n->inputs[0] = a;
        n->inputs[1] = b;
        n->visited = 0;
        Tensor_attach_gradients(out, n);
    }

    free(out_shape);
    return out;
}

int tensor_add_(Tensor *a, Tensor *b) {
    if (!a || !b) return 0;
    if (a->ndim != b->ndim) return 0;
    for (int d = 0; d < a->ndim; d++) if (a->shape[d] != b->shape[d]) return 0;

    /* Allow in-place only when autograd is disabled or both tensors do NOT require grad. */
    if (autograd_get_enabled() && (a->requires_grad || b->requires_grad)) return 0;

    int *idx = (int *)malloc(sizeof(int) * a->ndim);
    if (!idx) return 0;

    for (size_t i = 0; i < a->size; i++) {
        unravel_index((int)i, a->shape, a->ndim, idx);
        int off_a = tensor_offset(a, idx);
        int off_b = tensor_offset(b, idx);
        a->data[off_a] += b->data[off_b];
    }

    free(idx);
    return 1;
}

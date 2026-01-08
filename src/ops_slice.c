#include "ops_slice.h"
#include "autograd.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    AutogradNode base;
    int axis;
    int start;
    int slice_len;
    size_t tail;
    size_t outer;
    int contiguous;
} SliceCtx;

static int is_contiguous(const Tensor *t) {
    if (!t) return 0;
    int *exp = (int *)malloc(sizeof(int) * t->ndim);
    if (!exp) return 0;
    tensor_contiguous_strides(t->shape, t->ndim, exp);
    int ok = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->strides[i] != exp[i]) { ok = 0; break; }
    }
    free(exp);
    return ok;
}

static void slice_bwd(AutogradNode *node) {
    SliceCtx *ctx = (SliceCtx *)node;
    Tensor *out = node->out;
    Tensor *a = node->a;
    if (!out || !out->grad || !a || !a->requires_grad) return;

    if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
    if (!a->grad) return;

    int axis = ctx->axis;
    int start = ctx->start;
    int slice_len = ctx->slice_len;

    if (ctx->contiguous) {
        for (size_t o = 0; o < ctx->outer; o++) {
            size_t off_out = o * (size_t)slice_len * ctx->tail;
            size_t off_a = o * (size_t)a->shape[axis] * ctx->tail + (size_t)start * ctx->tail;
            for (size_t i = 0; i < (size_t)slice_len * ctx->tail; i++) {
                a->grad->data[off_a + i] += out->grad->data[off_out + i];
            }
        }
        return;
    }

    int *idx = (int *)malloc(sizeof(int) * a->ndim);
    if (!idx) return;

    for (size_t i = 0; i < out->size; i++) {
        unravel_index((int)i, out->shape, out->ndim, idx);
        idx[axis] += start;
        int off_a = tensor_offset(a->grad, idx);
        a->grad->data[off_a] += out->grad->data[i];
    }

    free(idx);
}

Tensor *tensor_slice(Tensor *a, int axis, int start, int end) {
    if (!a) return NULL;
    if (axis < 0) axis += a->ndim;
    if (axis < 0 || axis >= a->ndim) return NULL;

    int axis_len = a->shape[axis];
    if (start < 0) start += axis_len;
    if (end < 0) end += axis_len;
    if (start < 0 || end < 0 || start > end || end > axis_len) return NULL;

    int slice_len = end - start;
    if (slice_len <= 0) return NULL;

    int *out_shape = (int *)malloc(sizeof(int) * a->ndim);
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    out_shape[axis] = slice_len;

    Tensor *out = tensor_new(a->ndim, out_shape);
    free(out_shape);
    if (!out) return NULL;
    out->requires_grad = a->requires_grad;
    out->device = a->device;
    out->dtype = a->dtype;
    tensor_set_dtype(out, a->dtype);

    size_t tail = 1;
    for (int i = axis + 1; i < a->ndim; i++) tail *= (size_t)a->shape[i];
    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= (size_t)a->shape[i];

    int contiguous = is_contiguous(a);

    if (contiguous) {
        for (size_t o = 0; o < outer; o++) {
            size_t off_out = o * (size_t)slice_len * tail;
            size_t off_a = o * (size_t)a->shape[axis] * tail + (size_t)start * tail;
            memcpy(out->data + off_out, a->data + off_a, sizeof(float) * (size_t)slice_len * tail);
        }
    } else {
        int *idx_out = (int *)malloc(sizeof(int) * out->ndim);
        int *idx_in = (int *)malloc(sizeof(int) * a->ndim);
        if (!idx_out || !idx_in) { free(idx_out); free(idx_in); return out; }
        for (size_t i = 0; i < out->size; i++) {
            unravel_index((int)i, out->shape, out->ndim, idx_out);
            for (int d = 0; d < a->ndim; d++) idx_in[d] = idx_out[d];
            idx_in[axis] += start;
            int off_a = tensor_offset(a, idx_in);
            out->data[i] = a->data[off_a];
        }
        free(idx_out);
        free(idx_in);
    }

    if (out->requires_grad) {
        SliceCtx *ctx = (SliceCtx *)malloc(sizeof(SliceCtx));
        if (ctx) {
            memset(ctx, 0, sizeof(*ctx));
            ctx->base.out = out;
            ctx->base.a = a;
            ctx->base.b = NULL;
            ctx->base.n_inputs = 1;
            ctx->base.inputs = (Tensor **)malloc(sizeof(Tensor *));
            if (ctx->base.inputs) ctx->base.inputs[0] = a;
            ctx->base.backward = slice_bwd;
            ctx->axis = axis;
            ctx->start = start;
            ctx->slice_len = slice_len;
            ctx->tail = tail;
            ctx->outer = outer;
            ctx->contiguous = contiguous;
            Tensor_attach_gradients(out, (AutogradNode *)ctx);
        }
    }

    return out;
}

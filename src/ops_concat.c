#include "ops_concat.h"
#include "autograd.h"
#include "ops_reshape.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    AutogradNode base;
    int axis;
    int size_a;
    size_t tail;
} ConcatCtx;

static void concat_bwd(AutogradNode *node) {
    ConcatCtx *ctx = (ConcatCtx *)node;
    Tensor *out = node->out;
    Tensor *a = node->a;
    Tensor *b = node->b;
    if (!out || !out->grad) return;
    int axis = ctx->axis;
    size_t tail = ctx->tail;
    int size_a = ctx->size_a;
    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= (size_t)out->shape[i];

    if (a && a->requires_grad) {
        if (!a->grad) a->grad = tensor_zeros(a->ndim, a->shape);
        for (size_t o = 0; o < outer; o++) {
            size_t off_out = o * (size_t)out->shape[axis] * tail;
            size_t off_a = o * (size_t)size_a * tail;
            memcpy(a->grad->data + off_a, out->grad->data + off_out, sizeof(float) * (size_t)size_a * tail);
        }
    }
    if (b && b->requires_grad) {
        if (!b->grad) b->grad = tensor_zeros(b->ndim, b->shape);
        for (size_t o = 0; o < outer; o++) {
            size_t off_out = o * (size_t)out->shape[axis] * tail + (size_t)size_a * tail;
            size_t off_b = o * (size_t)b->shape[axis] * tail;
            memcpy(b->grad->data + off_b, out->grad->data + off_out, sizeof(float) * (size_t)b->shape[axis] * tail);
        }
    }
}

Tensor *tensor_concat(Tensor *a, Tensor *b, int axis) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    if (axis < 0 || axis >= a->ndim) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (i == axis) continue;
        if (a->shape[i] != b->shape[i]) return NULL;
    }
    if (a->dtype != b->dtype) return NULL;

    int *out_shape = (int *)malloc(sizeof(int) * a->ndim);
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    out_shape[axis] = a->shape[axis] + b->shape[axis];
    Tensor *out = tensor_new(a->ndim, out_shape);
    free(out_shape);
    if (!out) return NULL;
    out->requires_grad = (a->requires_grad || b->requires_grad);
    out->dtype = a->dtype;
    tensor_set_dtype(out, a->dtype);

    size_t tail = 1;
    for (int i = axis + 1; i < a->ndim; i++) tail *= (size_t)a->shape[i];
    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= (size_t)a->shape[i];

    for (size_t o = 0; o < outer; o++) {
        size_t off_out = o * (size_t)out->shape[axis] * tail;
        size_t off_a = o * (size_t)a->shape[axis] * tail;
        size_t off_b = o * (size_t)b->shape[axis] * tail;
        memcpy(out->data + off_out, a->data + off_a, sizeof(float) * (size_t)a->shape[axis] * tail);
        memcpy(out->data + off_out + (size_t)a->shape[axis] * tail, b->data + off_b, sizeof(float) * (size_t)b->shape[axis] * tail);
    }

    if (out->requires_grad) {
        ConcatCtx *ctx = (ConcatCtx *)malloc(sizeof(ConcatCtx));
        if (ctx) {
            memset(ctx, 0, sizeof(*ctx));
            ctx->base.out = out;
            ctx->base.a = a;
            ctx->base.b = b;
            ctx->base.n_inputs = 2;
            ctx->base.inputs = (Tensor **)malloc(sizeof(Tensor *) * 2);
            if (ctx->base.inputs) { ctx->base.inputs[0] = a; ctx->base.inputs[1] = b; }
            ctx->base.backward = concat_bwd;
            ctx->axis = axis;
            ctx->size_a = a->shape[axis];
            ctx->tail = tail;
            Tensor_attach_gradients(out, (AutogradNode *)ctx);
        }
    }

    return out;
}

Tensor *tensor_stack(Tensor *a, Tensor *b, int axis) {
    if (!a || !b) return NULL;
    if (a->ndim != b->ndim) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }
    if (axis < 0) axis = 0;
    if (axis > a->ndim) axis = a->ndim;

    Tensor *a_view = tensor_unsqueeze(a, axis);
    Tensor *b_view = tensor_unsqueeze(b, axis);
    Tensor *out = tensor_concat(a_view, b_view, axis);
    if (a_view && a_view != a) tensor_free(a_view);
    if (b_view && b_view != b) tensor_free(b_view);
    return out;
}

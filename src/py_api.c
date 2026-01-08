#include "tensor.h"
#include "autograd.h"
#include "ops_add.h"
#include "ops_mul.h"
#include "ops_div.h"
#include "ops_exp.h"
#include "ops_log.h"
#include "ops_softmax.h"
#include "ops_matmul.h"
#include "ops_reduce.h"
#include "ops_abs.h"
#include "ops_neg.h"
#include "ops_clamp.h"
#include "ops_reshape.h"
#include "ops_sub.h"
#include "ops_sqrt.h"
#include "ops_logsumexp.h"
#include "ops_nan.h"
#include "ops_clamp_min.h"
#include "ops_embedding.h"
#include "ops_conv2d.h"
#include "ops_bce.h"
#include "ops_crossentropy.h"
#include "ops_dropout.h"
#include "ops_batchnorm.h"
#include "ops_concat.h"
#include "ops_pad.h"
#include "ops_slice.h"
#include "profiler.h"
#include "backend.h"
#include <stdlib.h>
#include <stddef.h>

/* Optimizer bindings */
#include "optim.h"
/* Pooling */
#include "ops_maxpool.h"
#include "ops_avgpool.h"

/* Minimal C API exported for Python bindings (ctypes-friendly) */

Tensor *py_tensor_new(int ndim, const int *shape) { return tensor_new(ndim, shape); }
void py_tensor_free(Tensor *t) { if (t) tensor_free(t); }
int py_tensor_save(Tensor *t, const char *path) { return tensor_save(t, path); }
Tensor *py_tensor_load(const char *path) { return tensor_load(path); }
void py_tensor_print(Tensor *t) { tensor_print(t); }

float *py_tensor_data_ptr(Tensor *t) { return t ? t->data : NULL; }
float *py_tensor_grad_ptr(Tensor *t) { return (t && t->grad) ? t->grad->data : NULL; }
int py_tensor_ndim(Tensor *t) { return t ? t->ndim : 0; }
int py_tensor_shape_get(Tensor *t, int idx) { return (t && idx >= 0 && idx < t->ndim) ? t->shape[idx] : 0; }

Tensor *py_tensor_new_like(Tensor *t, int requires_grad) { return tensor_new_like(t, requires_grad); }
Tensor *py_sum(Tensor *t) { return tensor_sum(t); }
int py_tensor_get_requires_grad(Tensor *t) { return tensor_get_requires_grad(t); }
void py_tensor_set_requires_grad(Tensor *t, int req) { tensor_set_requires_grad(t, req); }

Tensor *py_add(Tensor *a, Tensor *b) { return tensor_add(a, b); }
Tensor *py_mul(Tensor *a, Tensor *b) { return tensor_mul(a, b); }
Tensor *py_matmul(Tensor *a, Tensor *b) {
	Backend *bk = backend_get();
	if (bk && bk->matmul) {
		Tensor *r = bk->matmul(a, b);
		if (r) return r;
	}
	return tensor_matmul(a, b);
}
Tensor *py_div(Tensor *a, Tensor *b) { return tensor_div(a, b); }
Tensor *py_exp(Tensor *a) { return tensor_exp(a); }
Tensor *py_log(Tensor *a) { return tensor_log(a); }
Tensor *py_log_softmax(Tensor *a) { return tensor_log_softmax(a); }
Tensor *py_softmax(Tensor *a) { return tensor_softmax(a); }
Tensor *py_sub(Tensor *a, Tensor *b) { return tensor_sub(a, b); }
Tensor *py_abs(Tensor *a) { return tensor_abs(a); }
Tensor *py_neg(Tensor *a) { return tensor_neg(a); }
Tensor *py_clamp(Tensor *a, float minv, float maxv) { return tensor_clamp(a, minv, maxv); }
Tensor *py_argmax(Tensor *a) { return tensor_argmax(a); }
Tensor *py_argmin(Tensor *a) { return tensor_argmin(a); }
Tensor *py_squeeze(Tensor *a, int dim) { return tensor_squeeze(a, dim); }
Tensor *py_unsqueeze(Tensor *a, int dim) { return tensor_unsqueeze(a, dim); }
Tensor *py_sqrt(Tensor *a) { return tensor_sqrt(a); }
Tensor *py_logsumexp(Tensor *a) { return tensor_logsumexp(a); }
Tensor *py_has_nan_inf(Tensor *a) { return tensor_has_nan_inf(a); }
Tensor *py_clamp_min(Tensor *a, float minv) { return tensor_clamp_min(a, minv); }
Tensor *py_embedding(Tensor *weights, Tensor *indices) { return tensor_embedding(weights, indices); }
Tensor *py_conv2d(Tensor *input, Tensor *weight, Tensor *bias) {
	Backend *bk = backend_get();
	if (bk && bk->conv2d) {
		Tensor *r = bk->conv2d(input, weight, bias);
		if (r) return r;
	}
	return tensor_conv2d(input, weight, bias);
}
Tensor *py_bce_loss(Tensor *pred, Tensor *target, Tensor *weight, int logits, int reduction) { return tensor_bce_loss(pred, target, weight, logits, reduction); }
Tensor *py_cross_entropy_logits(Tensor *logits, Tensor *target, Tensor *weight, int reduction) { return tensor_cross_entropy_logits(logits, target, weight, reduction); }
Tensor *py_reshape(Tensor *a, int new_ndim, const int *new_shape) { return tensor_reshape(a, new_ndim, new_shape); }
Tensor *py_maxpool2d(Tensor *x, int kernel_size) { return tensor_maxpool2d(x, kernel_size); }
Tensor *py_avgpool2d(Tensor *x, int kernel_size) { return tensor_avgpool2d(x, kernel_size); }

SGD *py_sgd_create(void **params, int n_params, float lr, float momentum, float weight_decay) {
	/* params: array of Tensor* pointers provided by Python layer */
	Tensor **arr = (Tensor **)params;
	return sgd_create(arr, n_params, lr, momentum, weight_decay);
}

void py_sgd_step(SGD *opt, float clip_norm) { sgd_step(opt, clip_norm); }

void py_sgd_zero_grad(SGD *opt) { sgd_zero_grad(opt); }

void py_sgd_set_lr(SGD *opt, float lr) { sgd_set_lr(opt, lr); }
float py_sgd_get_lr(SGD *opt) { return sgd_get_lr(opt); }
void py_sgd_set_param_lr(SGD *opt, int idx, float lr) { sgd_set_param_lr(opt, idx, lr); }
void py_sgd_set_param_weight_decay(SGD *opt, int idx, float wd) { sgd_set_param_weight_decay(opt, idx, wd); }
int py_sgd_save_state(SGD *opt, const char *path) { return sgd_save_state(opt, path); }
int py_sgd_load_state(SGD *opt, const char *path) { return sgd_load_state(opt, path); }

void py_sgd_free(SGD *opt) { sgd_free(opt); }

Adam *py_adam_create(void **params, int n_params, float lr, float beta1, float beta2, float eps, float weight_decay) {
	Tensor **arr = (Tensor **)params;
	return adam_create(arr, n_params, lr, beta1, beta2, eps, weight_decay);
}

void py_adam_step(Adam *opt, float clip_norm) { adam_step(opt, clip_norm); }
void py_adam_zero_grad(Adam *opt) { adam_zero_grad(opt); }
void py_adam_set_lr(Adam *opt, float lr) { adam_set_lr(opt, lr); }
float py_adam_get_lr(Adam *opt) { return adam_get_lr(opt); }
int py_adam_save_state(Adam *opt, const char *path) { return adam_save_state(opt, path); }
int py_adam_load_state(Adam *opt, const char *path) { return adam_load_state(opt, path); }
void py_adam_free(Adam *opt) { adam_free(opt); }

RMSProp *py_rmsprop_create(void **params, int n_params, float lr, float alpha, float eps, float weight_decay) {
	Tensor **arr = (Tensor **)params;
	return rmsprop_create(arr, n_params, lr, alpha, eps, weight_decay);
}

void py_rmsprop_step(RMSProp *opt, float clip_norm) { rmsprop_step(opt, clip_norm); }
void py_rmsprop_zero_grad(RMSProp *opt) { rmsprop_zero_grad(opt); }
void py_rmsprop_set_lr(RMSProp *opt, float lr) { rmsprop_set_lr(opt, lr); }
float py_rmsprop_get_lr(RMSProp *opt) { return rmsprop_get_lr(opt); }
int py_rmsprop_save_state(RMSProp *opt, const char *path) { return rmsprop_save_state(opt, path); }
int py_rmsprop_load_state(RMSProp *opt, const char *path) { return rmsprop_load_state(opt, path); }
void py_rmsprop_free(RMSProp *opt) { rmsprop_free(opt); }

Tensor *py_dropout(Tensor *x, float p, int training) { return tensor_dropout(x, p, training); }

Tensor *py_batchnorm(Tensor *x, Tensor *gamma, Tensor *beta, float eps, int training, Tensor *running_mean, Tensor *running_var, float momentum) {
	return tensor_batchnorm(x, gamma, beta, eps, training, running_mean, running_var, momentum);
}

Tensor *py_transpose(Tensor *x) { return tensor_transpose(x); }
Tensor *py_permute(Tensor *x, int ndim, const int *order) { return tensor_permute(x, order, ndim); }
Tensor *py_concat(Tensor *a, Tensor *b, int axis) { return tensor_concat(a, b, axis); }
Tensor *py_stack(Tensor *a, Tensor *b, int axis) { return tensor_stack(a, b, axis); }
Tensor *py_pad2d(Tensor *x, int pad_h, int pad_w, float value) { return tensor_pad2d(x, pad_h, pad_w, value); }
Tensor *py_slice(Tensor *x, int axis, int start, int end) { return tensor_slice(x, axis, start, end); }

char *py_profiler_get_summary() { return profiler_get_summary(); }

/* free helper for strings allocated by profiler */
void py_free_str(char *s) { if (s) free(s); }

const char *py_backend_get_name() { return backend_name(); }
int py_backend_set_by_name(const char *name) { return backend_set_by_name(name); }

/* Backward/grad helpers */
void py_zero_grad(Tensor *t) { tensor_zero_grad(t); }
void py_backward(Tensor *t) { tensor_backward(t); }
void py_backward_with_retain(Tensor *t, int retain) { tensor_backward_with_retain(t, retain); }

void py_autograd_set_enabled(int enabled) { autograd_set_enabled(enabled); }
int py_autograd_get_enabled() { return autograd_get_enabled(); }
void py_autograd_set_retain_default(int retain) { autograd_set_retain_default(retain); }
int py_autograd_get_retain_default() { return autograd_get_retain_default(); }

/* push/pop helpers to support nested "no_grad" contexts safely */
int py_autograd_push_disabled() { int prev = autograd_get_enabled(); autograd_set_enabled(0); return prev; }
void py_autograd_pop(int prev) { autograd_set_enabled(prev); }

const char *py_autograd_to_dot(Tensor *t) { return autograd_to_dot(t); }

/* DType / Device helpers */
int py_tensor_set_dtype(Tensor *t, int dtype) { return tensor_set_dtype(t, dtype); }
int py_tensor_get_device(Tensor *t) { return tensor_get_device(t); }
void py_tensor_set_device(Tensor *t, int device) { tensor_set_device(t, device); }

/* Copy tensor to a target device. For now this performs a CPU-backed copy and
 * marks the result's device field. This is a simple helper for testing device
 * round-trips; real GPU kernels are out of scope for the stub backend.
 */
Tensor *py_tensor_to_device(Tensor *t, int device) {
	if (!t) return NULL;
	Tensor *out = tensor_new_like(t, t->requires_grad);
	if (!out) return NULL;
	/* copy elementwise using float view (supports float32 storage); more
	   complete dtype-aware copying can be added later */
	for (size_t i = 0; i < t->size; i++) {
		tensor_set_f32_at(out, i, tensor_get_f32_at(t, i));
	}
	out->device = device;
	return out;
}

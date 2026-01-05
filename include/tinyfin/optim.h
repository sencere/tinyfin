#ifndef CTORCH_OPTIM_H
#define CTORCH_OPTIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

typedef struct SGD {
	Tensor **params;
	int n_params;
	float lr;
    float momentum;
    float weight_decay;
    Tensor **velocity; /* per-param momentum buffers, optional */
    float *lr_per_param; /* optional per-param overrides */
    float *wd_per_param; /* optional per-param overrides */
} SGD;

SGD *sgd_create(Tensor **params, int n_params, float lr, float momentum, float weight_decay);
void sgd_step(SGD *opt, float clip_norm);
void sgd_zero_grad(SGD *opt);
void sgd_set_lr(SGD *opt, float lr);
float sgd_get_lr(SGD *opt);
void sgd_set_param_lr(SGD *opt, int idx, float lr);
void sgd_set_param_weight_decay(SGD *opt, int idx, float wd);
int sgd_save_state(SGD *opt, const char *path);
int sgd_load_state(SGD *opt, const char *path);
void sgd_free(SGD *opt);

typedef struct Adam {
	Tensor **params;
	int n_params;
	float lr;
	float beta1, beta2, eps;
	int t;
	Tensor **m; /* first moment buffers */
	Tensor **v; /* second moment buffers */
} Adam;

Adam *adam_create(Tensor **params, int n_params, float lr, float beta1, float beta2, float eps, float weight_decay);
void adam_step(Adam *opt, float clip_norm);
void adam_zero_grad(Adam *opt);
void adam_set_lr(Adam *opt, float lr);
float adam_get_lr(Adam *opt);
int adam_save_state(Adam *opt, const char *path);
int adam_load_state(Adam *opt, const char *path);
void adam_free(Adam *opt);

typedef struct RMSProp {
	Tensor **params;
	int n_params;
	float lr;
	float alpha; /* decay */
	float eps;
	Tensor **s; /* running avg of squared grads */
	float weight_decay;
} RMSProp;

RMSProp *rmsprop_create(Tensor **params, int n_params, float lr, float alpha, float eps, float weight_decay);
void rmsprop_step(RMSProp *opt, float clip_norm);
void rmsprop_zero_grad(RMSProp *opt);
void rmsprop_set_lr(RMSProp *opt, float lr);
float rmsprop_get_lr(RMSProp *opt);
int rmsprop_save_state(RMSProp *opt, const char *path);
int rmsprop_load_state(RMSProp *opt, const char *path);
void rmsprop_free(RMSProp *opt);

#endif

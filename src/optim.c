#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "optim.h"
#include "tensor.h"

static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) exit(1);
  return p;
}

SGD *sgd_create(Tensor **params, int n_params, float lr, float momentum, float weight_decay) {
  SGD *o = (SGD *)xmalloc(sizeof(SGD));
  o->params = (Tensor **)xmalloc(sizeof(Tensor*) * (size_t)n_params);
  memcpy(o->params, params, sizeof(Tensor*) * (size_t)n_params);
  o->n_params = n_params;
  o->lr = lr;
  o->momentum = momentum;
  o->weight_decay = weight_decay;
  o->lr_per_param = NULL;
  o->wd_per_param = NULL;
  if (momentum > 0.0f) {
    o->velocity = (Tensor **)xmalloc(sizeof(Tensor*) * (size_t)n_params);
    for (int i = 0; i < n_params; i++) {
      Tensor *p = params[i];
      if (p) o->velocity[i] = tensor_zeros(p->ndim, p->shape);
      else o->velocity[i] = NULL;
    }
  } else {
    o->velocity = NULL;
  }
  return o;
}

void sgd_step(SGD *opt, float clip_norm) {
  float total_sq = 0.0f;
  if (clip_norm > 0.0f) {
    for (size_t p = 0; p < (size_t)opt->n_params; p++) {
      Tensor *t = opt->params[p];
      if (!t || !t->grad) continue;
      for (size_t i = 0; i < t->size; i++) {
        float g = t->grad->data[i];
        total_sq += g * g;
      }
    }
  }
  float scale = 1.0f;
  if (clip_norm > 0.0f && total_sq > 0.0f) {
    float norm = sqrtf(total_sq);
    if (norm > clip_norm) scale = clip_norm / norm;
  }
  for (size_t p = 0; p < (size_t)opt->n_params; p++) {
    Tensor *t = opt->params[p];
    if (!t || !t->grad) continue;
    float wd = opt->wd_per_param ? opt->wd_per_param[p] : opt->weight_decay;
    float lr = opt->lr_per_param ? opt->lr_per_param[p] : opt->lr;
    if (opt->momentum > 0.0f && opt->velocity) {
      Tensor *v = opt->velocity[p];
      for (size_t i = 0; i < t->size; i++) {
        float g = t->grad->data[i] * scale;
        if (wd != 0.0f) g += wd * t->data[i];
        v->data[i] = opt->momentum * v->data[i] + g;
        t->data[i] -= lr * v->data[i];
      }
    } else {
      for (size_t i = 0; i < t->size; i++) {
        float g = t->grad->data[i] * scale;
        if (wd != 0.0f) g += wd * t->data[i];
        t->data[i] -= lr * g;
      }
    }
  }
}

void sgd_zero_grad(SGD *opt) {
  for (size_t p = 0; p < (size_t)opt->n_params; p++) {
    Tensor *t = opt->params[p];
    if (!t || !t->grad) continue;
    memset(t->grad->data, 0, sizeof(float) * (size_t)t->size);
  }
}

void sgd_set_lr(SGD *opt, float lr) {
  if (!opt) return;
  opt->lr = lr;
}

float sgd_get_lr(SGD *opt) {
  if (!opt) return 0.0f;
  return opt->lr;
}

void sgd_set_param_lr(SGD *opt, int idx, float lr) {
  if (!opt || idx < 0 || idx >= opt->n_params) return;
  if (!opt->lr_per_param) {
    opt->lr_per_param = (float*)calloc((size_t)opt->n_params, sizeof(float));
    for (int i=0;i<opt->n_params;i++) opt->lr_per_param[i] = opt->lr;
  }
  opt->lr_per_param[idx] = lr;
}

void sgd_set_param_weight_decay(SGD *opt, int idx, float wd) {
  if (!opt || idx < 0 || idx >= opt->n_params) return;
  if (!opt->wd_per_param) {
    opt->wd_per_param = (float*)calloc((size_t)opt->n_params, sizeof(float));
    for (int i=0;i<opt->n_params;i++) opt->wd_per_param[i] = opt->weight_decay;
  }
  opt->wd_per_param[idx] = wd;
}

int sgd_save_state(SGD *opt, const char *path) {
  if (!opt || !path) return 0;
  FILE *f = fopen(path, "wb");
  if (!f) return 0;
  const uint32_t magic = 0x53474431; /* 'SGD1' */
  fwrite(&magic, sizeof(magic), 1, f);
  fwrite(&opt->n_params, sizeof(int), 1, f);
  fwrite(&opt->lr, sizeof(float), 1, f);
  fwrite(&opt->momentum, sizeof(float), 1, f);
  fwrite(&opt->weight_decay, sizeof(float), 1, f);
  /* per-param flags */
  uint8_t has_lr = opt->lr_per_param ? 1 : 0;
  uint8_t has_wd = opt->wd_per_param ? 1 : 0;
  fwrite(&has_lr, sizeof(uint8_t), 1, f);
  fwrite(&has_wd, sizeof(uint8_t), 1, f);
  if (has_lr) fwrite(opt->lr_per_param, sizeof(float), (size_t)opt->n_params, f);
  if (has_wd) fwrite(opt->wd_per_param, sizeof(float), (size_t)opt->n_params, f);
  /* velocities */
  uint8_t has_vel = (opt->momentum > 0.0f && opt->velocity) ? 1 : 0;
  fwrite(&has_vel, sizeof(uint8_t), 1, f);
  if (has_vel) {
    for (int i=0;i<opt->n_params;i++) {
      Tensor *v = opt->velocity[i];
      int ndim = v ? v->ndim : 0;
      fwrite(&ndim, sizeof(int), 1, f);
      if (ndim == 0 || !v) continue;
      fwrite(v->shape, sizeof(int), (size_t)ndim, f);
      fwrite(v->data, sizeof(float), v->size, f);
    }
  }
  fclose(f);
  return 1;
}

int sgd_load_state(SGD *opt, const char *path) {
  if (!opt || !path) return 0;
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  uint32_t magic=0; if (fread(&magic, sizeof(magic), 1, f) != 1) { fclose(f); return 0; }
  if (magic != 0x53474431) { fclose(f); return 0; }
  int n_params=0; if (fread(&n_params, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
  if (n_params != opt->n_params) { fclose(f); return 0; }
  if (fread(&opt->lr, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
  if (fread(&opt->momentum, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
  if (fread(&opt->weight_decay, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
  uint8_t has_lr=0, has_wd=0;
  if (fread(&has_lr, sizeof(uint8_t), 1, f) != 1) { fclose(f); return 0; }
  if (fread(&has_wd, sizeof(uint8_t), 1, f) != 1) { fclose(f); return 0; }
  if (has_lr) {
    if (!opt->lr_per_param) opt->lr_per_param = (float*)calloc((size_t)opt->n_params, sizeof(float));
    if (fread(opt->lr_per_param, sizeof(float), (size_t)opt->n_params, f) != (size_t)opt->n_params) { fclose(f); return 0; }
  }
  if (has_wd) {
    if (!opt->wd_per_param) opt->wd_per_param = (float*)calloc((size_t)opt->n_params, sizeof(float));
    if (fread(opt->wd_per_param, sizeof(float), (size_t)opt->n_params, f) != (size_t)opt->n_params) { fclose(f); return 0; }
  }
  uint8_t has_vel=0; if (fread(&has_vel, sizeof(uint8_t), 1, f) != 1) { fclose(f); return 0; }
  if (has_vel) {
    if (!opt->velocity) {
      opt->velocity = (Tensor**)calloc((size_t)opt->n_params, sizeof(Tensor*));
    }
    for (int i=0;i<opt->n_params;i++) {
      int ndim=0; if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
      if (ndim == 0) continue;
      int *shape = (int*)xmalloc(sizeof(int)*(size_t)ndim);
      if (fread(shape, sizeof(int), (size_t)ndim, f) != (size_t)ndim) { free(shape); fclose(f); return 0; }
      Tensor *v = opt->velocity[i];
      if (!v) {
        v = tensor_zeros(ndim, shape);
        opt->velocity[i] = v;
      }
      free(shape);
      if (fread(v->data, sizeof(float), v->size, f) != v->size) { fclose(f); return 0; }
    }
  }
  fclose(f);
  return 1;
}
void sgd_free(SGD *opt) {
  if (!opt) return;
  if (opt->velocity) {
    for (int i = 0; i < opt->n_params; i++) {
      if (opt->velocity[i]) tensor_free(opt->velocity[i]);
    }
    free(opt->velocity);
  }
  if (opt->params) free(opt->params);
  free(opt);
}

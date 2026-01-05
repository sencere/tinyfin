#include "optim.h"
#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

Adam *adam_create(Tensor **params, int n_params, float lr, float beta1, float beta2, float eps, float weight_decay) {
    Adam *o = malloc(sizeof(Adam));
    if (!o) return NULL;
    (void)weight_decay;
    o->params = malloc(sizeof(Tensor*) * (size_t)n_params);
    if (!o->params) { free(o); return NULL; }
    for (int i=0;i<n_params;i++) o->params[i] = params[i];
    o->n_params = n_params;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->eps = eps;
    o->t = 0;
    o->m = malloc(sizeof(Tensor*) * n_params);
    o->v = malloc(sizeof(Tensor*) * n_params);
    for (int i=0;i<n_params;i++) {
        Tensor *p = params[i];
        if (p) { o->m[i] = tensor_zeros(p->ndim, p->shape); o->v[i] = tensor_zeros(p->ndim, p->shape); }
        else { o->m[i]=NULL; o->v[i]=NULL; }
    }
    return o;
}

void adam_step(Adam *opt, float clip_norm) {
    if (!opt) return;
    float total_sq = 0.0f;
    if (clip_norm > 0.0f) {
        for (int i=0;i<opt->n_params;i++){
            Tensor *p = opt->params[i]; if (!p || !p->grad) continue;
            for (size_t k=0;k<p->size;k++){
                float g = p->grad->data[k];
                total_sq += g * g;
            }
        }
    }
    float scale = 1.0f;
    if (clip_norm > 0.0f && total_sq > 0.0f) {
        float norm = sqrtf(total_sq);
        if (norm > clip_norm) scale = clip_norm / norm;
    }
    opt->t += 1;
    for (int i=0;i<opt->n_params;i++){
        Tensor *p = opt->params[i]; if (!p || !p->grad) continue;
        Tensor *m = opt->m[i]; Tensor *v = opt->v[i];
        for (size_t k=0;k<p->size;k++){
            float g = p->grad->data[k] * scale;
            m->data[k] = opt->beta1 * m->data[k] + (1 - opt->beta1) * g;
            v->data[k] = opt->beta2 * v->data[k] + (1 - opt->beta2) * g * g;
            float m_hat = m->data[k] / (1 - powf(opt->beta1, (float)opt->t));
            float v_hat = v->data[k] / (1 - powf(opt->beta2, (float)opt->t));
            p->data[k] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}

void adam_zero_grad(Adam *opt) {
    if (!opt) return;
    for (int i=0;i<opt->n_params;i++){
        Tensor *p = opt->params[i]; if (p && p->grad) for (size_t k=0;k<p->grad->size;k++) p->grad->data[k]=0.0f;
    }
}

void adam_set_lr(Adam *opt, float lr){ if (opt) opt->lr = lr; }
float adam_get_lr(Adam *opt){ return opt ? opt->lr : 0.0f; }

int adam_save_state(Adam *opt, const char *path){
    if (!opt || !path) return 0;
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    const uint32_t magic = 0x41444d31; /* 'ADM1' */
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&opt->n_params, sizeof(int), 1, f);
    fwrite(&opt->lr, sizeof(float), 1, f);
    fwrite(&opt->beta1, sizeof(float), 1, f);
    fwrite(&opt->beta2, sizeof(float), 1, f);
    fwrite(&opt->eps, sizeof(float), 1, f);
    fwrite(&opt->t, sizeof(int), 1, f);
    for (int i=0;i<opt->n_params;i++){
        Tensor *m = opt->m[i]; Tensor *v = opt->v[i];
        int ndim = m ? m->ndim : 0;
        fwrite(&ndim, sizeof(int), 1, f);
        if (ndim==0 || !m || !v) continue;
        fwrite(m->shape, sizeof(int), (size_t)ndim, f);
        fwrite(m->data, sizeof(float), m->size, f);
        fwrite(v->data, sizeof(float), v->size, f);
    }
    fclose(f);
    return 1;
}

int adam_load_state(Adam *opt, const char *path){
    if (!opt || !path) return 0;
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    uint32_t magic=0; if (fread(&magic, sizeof(magic), 1, f) != 1) { fclose(f); return 0; }
    if (magic != 0x41444d31) { fclose(f); return 0; }
    int n_params=0; if (fread(&n_params, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (n_params != opt->n_params) { fclose(f); return 0; }
    if (fread(&opt->lr, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->beta1, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->beta2, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->eps, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->t, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    for (int i=0;i<opt->n_params;i++){
        int ndim=0; if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (ndim==0) continue;
        int *shape = (int*)malloc(sizeof(int)*(size_t)ndim);
        if (!shape) { fclose(f); return 0; }
        if (fread(shape, sizeof(int), (size_t)ndim, f) != (size_t)ndim) { free(shape); fclose(f); return 0; }
        Tensor *m = opt->m[i];
        if (!m) { m = tensor_zeros(ndim, shape); opt->m[i]=m; }
        Tensor *v = opt->v[i];
        if (!v) { v = tensor_zeros(ndim, shape); opt->v[i]=v; }
        free(shape);
        if (fread(m->data, sizeof(float), m->size, f) != m->size) { fclose(f); return 0; }
        if (fread(v->data, sizeof(float), v->size, f) != v->size) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

void adam_free(Adam *opt){
    if (!opt) return;
    if (opt->m){
        for (int i=0;i<opt->n_params;i++){
            if (opt->m[i]) tensor_free(opt->m[i]);
        }
        free(opt->m);
    }
    if (opt->v){
        for (int i=0;i<opt->n_params;i++){
            if (opt->v[i]) tensor_free(opt->v[i]);
        }
        free(opt->v);
    }
    if (opt->params) free(opt->params);
    free(opt);
}

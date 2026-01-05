#include "optim.h"
#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

RMSProp *rmsprop_create(Tensor **params, int n_params, float lr, float alpha, float eps, float weight_decay) {
    RMSProp *o = malloc(sizeof(RMSProp)); if (!o) return NULL;
    o->params = malloc(sizeof(Tensor*) * (size_t)n_params);
    if (!o->params) { free(o); return NULL; }
    for (int i=0;i<n_params;i++) o->params[i] = params[i];
    o->n_params = n_params; o->lr = lr; o->alpha = alpha; o->eps = eps; o->weight_decay = weight_decay;
    o->s = malloc(sizeof(Tensor*)*n_params);
    for (int i=0;i<n_params;i++) {
        Tensor *p = params[i];
        if (p) o->s[i] = tensor_zeros(p->ndim, p->shape);
        else o->s[i] = NULL;
    }
    return o;
}

void rmsprop_step(RMSProp *opt, float clip_norm) {
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
    for (int i=0;i<opt->n_params;i++){
        Tensor *p = opt->params[i]; if (!p || !p->grad) continue;
        Tensor *s = opt->s[i];
        for (size_t k=0;k<p->size;k++){
            float g = p->grad->data[k] * scale;
            if (opt->weight_decay != 0.0f) g += opt->weight_decay * p->data[k];
            s->data[k] = opt->alpha * s->data[k] + (1-opt->alpha) * g * g;
            p->data[k] -= opt->lr * g / (sqrtf(s->data[k]) + opt->eps);
        }
    }
}

void rmsprop_zero_grad(RMSProp *opt) { if (!opt) return; for (int i=0;i<opt->n_params;i++){ Tensor *p=opt->params[i]; if (p && p->grad) for (size_t k=0;k<p->grad->size;k++) p->grad->data[k]=0.0f; } }

void rmsprop_set_lr(RMSProp *opt, float lr){ if (opt) opt->lr = lr; }
float rmsprop_get_lr(RMSProp *opt){ return opt ? opt->lr : 0.0f; }

int rmsprop_save_state(RMSProp *opt, const char *path){
    if (!opt || !path) return 0;
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    const uint32_t magic = 0x524d5331; /* 'RMS1' */
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&opt->n_params, sizeof(int), 1, f);
    fwrite(&opt->lr, sizeof(float), 1, f);
    fwrite(&opt->alpha, sizeof(float), 1, f);
    fwrite(&opt->eps, sizeof(float), 1, f);
    fwrite(&opt->weight_decay, sizeof(float), 1, f);
    for (int i=0;i<opt->n_params;i++){
        Tensor *s = opt->s[i];
        int ndim = s ? s->ndim : 0;
        fwrite(&ndim, sizeof(int), 1, f);
        if (ndim==0 || !s) continue;
        fwrite(s->shape, sizeof(int), (size_t)ndim, f);
        fwrite(s->data, sizeof(float), s->size, f);
    }
    fclose(f);
    return 1;
}

int rmsprop_load_state(RMSProp *opt, const char *path){
    if (!opt || !path) return 0;
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    uint32_t magic=0; if (fread(&magic, sizeof(magic), 1, f) != 1) { fclose(f); return 0; }
    if (magic != 0x524d5331) { fclose(f); return 0; }
    int n_params=0; if (fread(&n_params, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (n_params != opt->n_params) { fclose(f); return 0; }
    if (fread(&opt->lr, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->alpha, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->eps, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&opt->weight_decay, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    for (int i=0;i<opt->n_params;i++){
        int ndim=0; if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (ndim==0) continue;
        int *shape = (int*)malloc(sizeof(int)*(size_t)ndim);
        if (!shape) { fclose(f); return 0; }
        if (fread(shape, sizeof(int), (size_t)ndim, f) != (size_t)ndim) { free(shape); fclose(f); return 0; }
        Tensor *s = opt->s[i];
        if (!s) { s = tensor_zeros(ndim, shape); opt->s[i]=s; }
        free(shape);
        if (fread(s->data, sizeof(float), s->size, f) != s->size) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

void rmsprop_free(RMSProp *opt){
    if (!opt) return;
    if (opt->s){
        for (int i=0;i<opt->n_params;i++){
            if (opt->s[i]) tensor_free(opt->s[i]);
        }
        free(opt->s);
    }
    if (opt->params) free(opt->params);
    free(opt);
}

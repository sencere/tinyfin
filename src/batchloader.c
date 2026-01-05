#include "batchloader.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

BatchLoader *batchloader_create(Tensor *data, Tensor *labels) {
    if (!data) return NULL;
    BatchLoader *b = malloc(sizeof(BatchLoader));
    if (!b) return NULL;
    b->data = data; b->labels = labels; b->N = (int)data->shape[0]; b->idx = 0;
    b->perm = malloc(sizeof(int) * b->N);
    for (int i=0;i<b->N;i++) b->perm[i]=i;
    return b;
}

void batchloader_shuffle(BatchLoader *b) {
    if (!b) return;
    for (int i=b->N-1;i>0;i--) {
        int j = rand() % (i+1);
        int t = b->perm[i]; b->perm[i]=b->perm[j]; b->perm[j]=t;
    }
    b->idx = 0;
}

/* Returns 1 if batch filled, 0 if no more data. Caller receives views (not copies). */
int batchloader_next(BatchLoader *b, int batch_size, Tensor **out_data, Tensor **out_labels) {
    if (!b) return 0;
    if (b->idx >= b->N) return 0;
    int remain = b->N - b->idx;
    int take = (batch_size < remain) ? batch_size : remain;
    /* Create a new Tensor view for the batch (shallow copy with data pointer offset) */
    int ndim = b->data->ndim;
    int *shape = malloc(sizeof(int)*ndim);
    memcpy(shape, b->data->shape, sizeof(int)*ndim);
    shape[0] = take;
    Tensor *d = tensor_new(ndim, shape);
    free(shape);
    /* fill d->data by copying selected rows */
    size_t row_size = 1; for (int i=1;i<ndim;i++) row_size *= b->data->shape[i];
    for (int i=0;i<take;i++) {
        int src = b->perm[b->idx + i];
        memcpy(&d->data[i*row_size], &b->data->data[src*row_size], sizeof(float)*row_size);
    }
    *out_data = d;
    if (b->labels) {
        int lndim = b->labels->ndim;
        int *lshape = malloc(sizeof(int)*lndim); memcpy(lshape, b->labels->shape, sizeof(int)*lndim); lshape[0]=take;
        Tensor *lab = tensor_new(lndim, lshape); free(lshape);
        size_t lrow = 1; for (int i=1;i<lndim;i++) lrow *= b->labels->shape[i];
        for (int i=0;i<take;i++) { int src = b->perm[b->idx + i]; memcpy(&lab->data[i*lrow], &b->labels->data[src*lrow], sizeof(float)*lrow); }
        *out_labels = lab;
    } else *out_labels = NULL;
    b->idx += take;
    return 1;
}

void batchloader_free(BatchLoader *b) { if (!b) return; if (b->perm) free(b->perm); free(b); }

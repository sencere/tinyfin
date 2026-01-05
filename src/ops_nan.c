#include "ops_nan.h"
#include <math.h>
#include <stdlib.h>

Tensor *tensor_has_nan_inf(Tensor *t){
    int shape[1] = {1};
    Tensor *out = tensor_new(1, shape);
    if (!out) return NULL;
    out->requires_grad = 0;
    float flag = 0.0f;
    for (size_t i=0;i<t->size;i++){
        float v = t->data[i];
        if (isnan(v) || isinf(v)) { flag = 1.0f; break; }
    }
    out->data[0] = flag;
    return out;
}

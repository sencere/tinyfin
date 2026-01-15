#include "graph.h"
#include "ops_add.h"
#include "ops_matmul.h"
#include "ops_activation.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void fill_range(Tensor *t, float start) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = start + (float)i;
    }
}

int main() {
    int x_shape[2] = {2, 3};
    int w_shape[2] = {3, 4};
    int b_shape[1] = {4};
    Tensor *x = tensor_new(2, x_shape);
    Tensor *w = tensor_new(2, w_shape);
    Tensor *b = tensor_new(1, b_shape);
    fill_range(x, 0.0f);
    fill_range(w, 0.5f);
    fill_range(b, -1.0f);

    graph_capture_begin();
    Tensor *y = tensor_matmul(x, w);
    Tensor *z = tensor_add(y, b);
    Tensor *r = tensor_relu(z);
    Tensor *outs[1] = {r};
    Graph *g = graph_capture_end(outs, 1);
    assert(g);

    GraphPlan *p = graph_compile(g);
    assert(p);

    Tensor *inputs[3] = {x, w, b};
    Tensor *out = graph_run(p, inputs, 3);
    assert(out);
    for (size_t i = 0; i < out->size; i++) {
        float diff = fabsf(out->data[i] - r->data[i]);
        assert(diff < 1e-5f);
    }

    int hit = 0;
    Tensor *out1 = graph_cache_run(g, inputs, 3, &hit);
    assert(out1);
    assert(hit == 0);
    Tensor *out2 = graph_cache_run(g, inputs, 3, &hit);
    assert(out2);
    assert(hit == 1);

    graph_plan_free(p);
    graph_free(g);
    tensor_free(out);
    tensor_free(out1);
    tensor_free(out2);
    tensor_free(r);
    tensor_free(z);
    tensor_free(y);
    tensor_free(b);
    tensor_free(w);
    tensor_free(x);

    printf("[test_graph_capture.c] Graph capture/compile/cache PASS\n");
    return 0;
}

#ifndef TINYFIN_GRAPH_H
#define TINYFIN_GRAPH_H

#include <stdint.h>
#include "tensor.h"

typedef enum {
    GRAPH_OP_INPUT = 0,
    GRAPH_OP_ADD,
    GRAPH_OP_SUB,
    GRAPH_OP_MUL,
    GRAPH_OP_DIV,
    GRAPH_OP_MATMUL,
    GRAPH_OP_RELU,
    GRAPH_OP_FUSED_ELEMENTWISE,
    GRAPH_OP_MATMUL_BIAS_RELU
} GraphOp;

typedef struct {
    int id;
    GraphOp op;
    int n_inputs;
    int *inputs;
    int ndim;
    int *shape;
    int dtype;
    int device;
} GraphNode;

typedef struct {
    GraphNode *nodes;
    int n_nodes;
    int cap_nodes;
    int *inputs;
    int n_inputs;
    int *outputs;
    int n_outputs;
} Graph;

typedef struct {
    GraphOp op;
    int id;
    int n_inputs;
    int *inputs;
    int n_chain;
    GraphNode *chain;
    int ndim;
    int *shape;
    int dtype;
    int device;
} GraphPlanNode;

typedef struct {
    GraphPlanNode *nodes;
    int n_nodes;
    int cap_nodes;
    int *inputs;
    int n_inputs;
    int *outputs;
    int n_outputs;
} GraphPlan;

void graph_capture_begin(void);
Graph *graph_capture_end(Tensor **outputs, int n_outputs);
int graph_capture_enabled(void);
void graph_capture_push_disabled(void);
void graph_capture_pop_disabled(void);
void graph_record_op(GraphOp op, Tensor *out, Tensor **inputs, int n_inputs);

Graph *graph_new(void);
void graph_free(Graph *g);
GraphPlan *graph_compile(Graph *g);
void graph_plan_free(GraphPlan *p);
Tensor *graph_run(GraphPlan *p, Tensor **inputs, int n_inputs);

uint64_t graph_signature(Graph *g);
GraphPlan *graph_cache_get(uint64_t key);
GraphPlan *graph_cache_put(uint64_t key, GraphPlan *p);
void graph_cache_clear(void);
Tensor *graph_cache_run(Graph *g, Tensor **inputs, int n_inputs, int *hit);

#endif

#include "graph.h"
#include "ops_add.h"
#include "ops_sub.h"
#include "ops_mul.h"
#include "ops_div.h"
#include "ops_matmul.h"
#include "ops_activation.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    Tensor *t;
    int id;
} TensorMap;

typedef struct {
    int enabled;
    int suppress;
    Graph *graph;
    TensorMap *map;
    int n_map;
    int cap_map;
} GraphRecorder;

static GraphRecorder g_recorder = {0};

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    return p;
}

static int *copy_shape(const Tensor *t) {
    if (!t || t->ndim <= 0) return NULL;
    int *shape = (int *)malloc(sizeof(int) * t->ndim);
    if (!shape) return NULL;
    memcpy(shape, t->shape, sizeof(int) * t->ndim);
    return shape;
}

Graph *graph_new(void) {
    Graph *g = (Graph *)xcalloc(1, sizeof(*g));
    return g;
}

static void graph_node_free(GraphNode *n) {
    if (!n) return;
    if (n->inputs) free(n->inputs);
    if (n->shape) free(n->shape);
}

void graph_free(Graph *g) {
    if (!g) return;
    for (int i = 0; i < g->n_nodes; i++) {
        graph_node_free(&g->nodes[i]);
    }
    if (g->nodes) free(g->nodes);
    if (g->inputs) free(g->inputs);
    if (g->outputs) free(g->outputs);
    free(g);
}

static int graph_add_node(Graph *g, GraphNode *node) {
    if (!g || !node) return -1;
    if (g->n_nodes + 1 > g->cap_nodes) {
        int new_cap = g->cap_nodes ? g->cap_nodes * 2 : 16;
        GraphNode *nn = (GraphNode *)realloc(g->nodes, sizeof(*g->nodes) * new_cap);
        if (!nn) return -1;
        g->nodes = nn;
        g->cap_nodes = new_cap;
    }
    g->nodes[g->n_nodes] = *node;
    g->nodes[g->n_nodes].id = g->n_nodes;
    return g->n_nodes++;
}

static void recorder_map_add(GraphRecorder *r, Tensor *t, int id) {
    if (r->n_map + 1 > r->cap_map) {
        int new_cap = r->cap_map ? r->cap_map * 2 : 16;
        TensorMap *nm = (TensorMap *)realloc(r->map, sizeof(*r->map) * new_cap);
        if (!nm) return;
        r->map = nm;
        r->cap_map = new_cap;
    }
    r->map[r->n_map].t = t;
    r->map[r->n_map].id = id;
    r->n_map++;
}

static int recorder_map_get(GraphRecorder *r, Tensor *t) {
    if (!r || !t) return -1;
    for (int i = 0; i < r->n_map; i++) {
        if (r->map[i].t == t) return r->map[i].id;
    }
    return -1;
}

static int recorder_add_input(GraphRecorder *r, Tensor *t) {
    GraphNode node;
    memset(&node, 0, sizeof(node));
    node.op = GRAPH_OP_INPUT;
    node.n_inputs = 0;
    node.inputs = NULL;
    node.ndim = t->ndim;
    node.shape = copy_shape(t);
    node.dtype = t->dtype;
    node.device = t->device;
    int id = graph_add_node(r->graph, &node);
    recorder_map_add(r, t, id);
    if (r->graph->n_inputs + 1 > 0) {
        int new_cap = r->graph->n_inputs + 1;
        int *nn = (int *)realloc(r->graph->inputs, sizeof(int) * new_cap);
        if (nn) {
            r->graph->inputs = nn;
            r->graph->inputs[r->graph->n_inputs++] = id;
        }
    }
    return id;
}

int graph_capture_enabled(void) {
    return g_recorder.enabled && g_recorder.suppress == 0;
}

void graph_capture_push_disabled(void) {
    g_recorder.suppress += 1;
}

void graph_capture_pop_disabled(void) {
    g_recorder.suppress -= 1;
    if (g_recorder.suppress < 0) g_recorder.suppress = 0;
}

void graph_capture_begin(void) {
    if (g_recorder.enabled) return;
    g_recorder.graph = graph_new();
    g_recorder.enabled = 1;
    g_recorder.suppress = 0;
    g_recorder.n_map = 0;
    g_recorder.cap_map = 0;
    if (g_recorder.map) {
        free(g_recorder.map);
        g_recorder.map = NULL;
    }
}

Graph *graph_capture_end(Tensor **outputs, int n_outputs) {
    if (!g_recorder.enabled) return NULL;
    Graph *g = g_recorder.graph;
    if (outputs && n_outputs > 0) {
        g->outputs = (int *)malloc(sizeof(int) * n_outputs);
        g->n_outputs = 0;
        for (int i = 0; i < n_outputs; i++) {
            Tensor *t = outputs[i];
            int id = recorder_map_get(&g_recorder, t);
            if (id < 0) {
                id = recorder_add_input(&g_recorder, t);
            }
            g->outputs[g->n_outputs++] = id;
        }
    }
    g_recorder.enabled = 0;
    g_recorder.graph = NULL;
    g_recorder.suppress = 0;
    g_recorder.n_map = 0;
    g_recorder.cap_map = 0;
    if (g_recorder.map) {
        free(g_recorder.map);
        g_recorder.map = NULL;
    }
    return g;
}

void graph_record_op(GraphOp op, Tensor *out, Tensor **inputs, int n_inputs) {
    if (!graph_capture_enabled()) return;
    if (!out || !inputs || n_inputs <= 0) return;
    GraphRecorder *r = &g_recorder;
    GraphNode node;
    memset(&node, 0, sizeof(node));
    node.op = op;
    node.n_inputs = n_inputs;
    node.inputs = (int *)malloc(sizeof(int) * n_inputs);
    if (!node.inputs) return;
    for (int i = 0; i < n_inputs; i++) {
        Tensor *t = inputs[i];
        int id = recorder_map_get(r, t);
        if (id < 0) {
            id = recorder_add_input(r, t);
        }
        node.inputs[i] = id;
    }
    node.ndim = out->ndim;
    node.shape = copy_shape(out);
    node.dtype = out->dtype;
    node.device = out->device;
    int id = graph_add_node(r->graph, &node);
    if (id >= 0) recorder_map_add(r, out, id);
}

static uint64_t fnv1a64_u64(uint64_t h, uint64_t v) {
    const uint64_t prime = 1099511628211ULL;
    h ^= v;
    h *= prime;
    return h;
}

uint64_t graph_signature(Graph *g) {
    if (!g) return 0;
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_u64(h, (uint64_t)g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        GraphNode *n = &g->nodes[i];
        h = fnv1a64_u64(h, (uint64_t)n->id);
        h = fnv1a64_u64(h, (uint64_t)n->op);
        h = fnv1a64_u64(h, (uint64_t)n->n_inputs);
        for (int j = 0; j < n->n_inputs; j++) {
            h = fnv1a64_u64(h, (uint64_t)n->inputs[j]);
        }
        h = fnv1a64_u64(h, (uint64_t)n->ndim);
        for (int d = 0; d < n->ndim; d++) {
            h = fnv1a64_u64(h, (uint64_t)n->shape[d]);
        }
        h = fnv1a64_u64(h, (uint64_t)n->dtype);
        h = fnv1a64_u64(h, (uint64_t)n->device);
    }
    h = fnv1a64_u64(h, (uint64_t)g->n_inputs);
    for (int i = 0; i < g->n_inputs; i++) h = fnv1a64_u64(h, (uint64_t)g->inputs[i]);
    h = fnv1a64_u64(h, (uint64_t)g->n_outputs);
    for (int i = 0; i < g->n_outputs; i++) h = fnv1a64_u64(h, (uint64_t)g->outputs[i]);
    return h;
}

typedef struct {
    uint64_t key;
    GraphPlan *plan;
} GraphCacheEntry;

static GraphCacheEntry *g_cache = NULL;
static int g_cache_len = 0;
static int g_cache_cap = 0;

GraphPlan *graph_cache_get(uint64_t key) {
    for (int i = 0; i < g_cache_len; i++) {
        if (g_cache[i].key == key) return g_cache[i].plan;
    }
    return NULL;
}

GraphPlan *graph_cache_put(uint64_t key, GraphPlan *p) {
    for (int i = 0; i < g_cache_len; i++) {
        if (g_cache[i].key == key) {
            g_cache[i].plan = p;
            return p;
        }
    }
    if (g_cache_len + 1 > g_cache_cap) {
        int new_cap = g_cache_cap ? g_cache_cap * 2 : 8;
        GraphCacheEntry *nn = (GraphCacheEntry *)realloc(g_cache, sizeof(*g_cache) * new_cap);
        if (!nn) return p;
        g_cache = nn;
        g_cache_cap = new_cap;
    }
    g_cache[g_cache_len].key = key;
    g_cache[g_cache_len].plan = p;
    g_cache_len++;
    return p;
}

void graph_cache_clear(void) {
    if (g_cache) free(g_cache);
    g_cache = NULL;
    g_cache_len = 0;
    g_cache_cap = 0;
}

static void plan_node_free(GraphPlanNode *n) {
    if (!n) return;
    if (n->inputs) free(n->inputs);
    if (n->chain) {
        for (int i = 0; i < n->n_chain; i++) {
            if (n->chain[i].inputs) free(n->chain[i].inputs);
            if (n->chain[i].shape) free(n->chain[i].shape);
        }
        free(n->chain);
    }
    if (n->shape) free(n->shape);
}

void graph_plan_free(GraphPlan *p) {
    if (!p) return;
    for (int i = 0; i < p->n_nodes; i++) plan_node_free(&p->nodes[i]);
    if (p->nodes) free(p->nodes);
    if (p->inputs) free(p->inputs);
    if (p->outputs) free(p->outputs);
    free(p);
}

static GraphPlanNode plan_node_from_graph(const GraphNode *n) {
    GraphPlanNode pn;
    memset(&pn, 0, sizeof(pn));
    pn.op = n->op;
    pn.id = n->id;
    pn.n_inputs = n->n_inputs;
    if (n->n_inputs > 0) {
        pn.inputs = (int *)malloc(sizeof(int) * n->n_inputs);
        if (pn.inputs) memcpy(pn.inputs, n->inputs, sizeof(int) * n->n_inputs);
    }
    pn.ndim = n->ndim;
    pn.shape = NULL;
    if (n->ndim > 0 && n->shape) {
        pn.shape = (int *)malloc(sizeof(int) * n->ndim);
        if (pn.shape) memcpy(pn.shape, n->shape, sizeof(int) * n->ndim);
    }
    pn.dtype = n->dtype;
    pn.device = n->device;
    return pn;
}

static GraphPlanNode plan_node_clone(const GraphPlanNode *n) {
    GraphPlanNode pn;
    memset(&pn, 0, sizeof(pn));
    pn.op = n->op;
    pn.id = n->id;
    pn.n_inputs = n->n_inputs;
    if (n->n_inputs > 0) {
        pn.inputs = (int *)malloc(sizeof(int) * n->n_inputs);
        if (pn.inputs) memcpy(pn.inputs, n->inputs, sizeof(int) * n->n_inputs);
    }
    pn.ndim = n->ndim;
    if (n->ndim > 0 && n->shape) {
        pn.shape = (int *)malloc(sizeof(int) * n->ndim);
        if (pn.shape) memcpy(pn.shape, n->shape, sizeof(int) * n->ndim);
    }
    pn.dtype = n->dtype;
    pn.device = n->device;
    return pn;
}

static int plan_add_node(GraphPlan *p, GraphPlanNode *node) {
    if (p->n_nodes + 1 > p->cap_nodes) {
        int new_cap = p->cap_nodes ? p->cap_nodes * 2 : 16;
        GraphPlanNode *nn = (GraphPlanNode *)realloc(p->nodes, sizeof(*p->nodes) * new_cap);
        if (!nn) return -1;
        p->nodes = nn;
        p->cap_nodes = new_cap;
    }
    p->nodes[p->n_nodes] = *node;
    return p->n_nodes++;
}

static GraphPlan *plan_new_from_graph(Graph *g) {
    GraphPlan *p = (GraphPlan *)xcalloc(1, sizeof(*p));
    if (!p) return NULL;
    p->inputs = NULL;
    if (g->n_inputs > 0) {
        p->inputs = (int *)malloc(sizeof(int) * g->n_inputs);
        if (p->inputs) {
            memcpy(p->inputs, g->inputs, sizeof(int) * g->n_inputs);
            p->n_inputs = g->n_inputs;
        }
    }
    p->outputs = NULL;
    if (g->n_outputs > 0) {
        p->outputs = (int *)malloc(sizeof(int) * g->n_outputs);
        if (p->outputs) {
            memcpy(p->outputs, g->outputs, sizeof(int) * g->n_outputs);
            p->n_outputs = g->n_outputs;
        }
    }
    return p;
}

static int is_elementwise(GraphOp op) {
    return op == GRAPH_OP_ADD || op == GRAPH_OP_SUB || op == GRAPH_OP_MUL ||
           op == GRAPH_OP_DIV || op == GRAPH_OP_RELU;
}

static GraphPlan *fuse_matmul_bias_relu(GraphPlan *p) {
    int n = p->n_nodes;
    GraphPlan *out = (GraphPlan *)xcalloc(1, sizeof(*out));
    if (!out) return p;
    out->inputs = NULL;
    out->outputs = NULL;
    if (p->n_inputs) {
        out->inputs = (int *)malloc(sizeof(int) * p->n_inputs);
        if (out->inputs) {
            memcpy(out->inputs, p->inputs, sizeof(int) * p->n_inputs);
            out->n_inputs = p->n_inputs;
        }
    }
    if (p->n_outputs) {
        out->outputs = (int *)malloc(sizeof(int) * p->n_outputs);
        if (out->outputs) {
            memcpy(out->outputs, p->outputs, sizeof(int) * p->n_outputs);
            out->n_outputs = p->n_outputs;
        }
    }

    GraphPlanNode **by_id = (GraphPlanNode **)xcalloc(n, sizeof(*by_id));
    int *users = (int *)xcalloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        GraphPlanNode *node = &p->nodes[i];
        if (node->id >= 0 && node->id < n) by_id[node->id] = node;
        for (int j = 0; j < node->n_inputs; j++) {
            int id = node->inputs[j];
            if (id >= 0 && id < n) users[id] += 1;
        }
    }

    int *skip = (int *)xcalloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        GraphPlanNode *node = &p->nodes[i];
        if (node->id >= 0 && node->id < n && skip[node->id]) continue;
        if (node->op != GRAPH_OP_RELU || node->n_inputs != 1) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        int add_id = node->inputs[0];
        if (add_id < 0 || add_id >= n) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        GraphPlanNode *add_node = by_id[add_id];
        if (!add_node || add_node->op != GRAPH_OP_ADD || users[add_id] != 1 || add_node->n_inputs != 2) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        int mm_id = add_node->inputs[0];
        int bias_id = add_node->inputs[1];
        GraphPlanNode *mm_node = NULL;
        if (mm_id >= 0 && mm_id < n && by_id[mm_id] && by_id[mm_id]->op == GRAPH_OP_MATMUL) {
            mm_node = by_id[mm_id];
        } else {
            mm_id = add_node->inputs[1];
            bias_id = add_node->inputs[0];
            if (mm_id >= 0 && mm_id < n && by_id[mm_id] && by_id[mm_id]->op == GRAPH_OP_MATMUL) {
                mm_node = by_id[mm_id];
            }
        }
        if (!mm_node || users[mm_id] != 1) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        GraphPlanNode *bias_node = (bias_id >= 0 && bias_id < n) ? by_id[bias_id] : NULL;
        if (!bias_node || mm_node->ndim != 2 || bias_node->ndim != 1) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        if (bias_node->shape && mm_node->shape) {
            if (bias_node->shape[0] != mm_node->shape[1]) {
                GraphPlanNode copy = plan_node_clone(node);
                plan_add_node(out, &copy);
                continue;
            }
        }
        GraphPlanNode fused;
        memset(&fused, 0, sizeof(fused));
        fused.op = GRAPH_OP_MATMUL_BIAS_RELU;
        fused.id = node->id;
        fused.n_inputs = mm_node->n_inputs + 1;
        fused.inputs = (int *)malloc(sizeof(int) * fused.n_inputs);
        if (fused.inputs) {
            for (int j = 0; j < mm_node->n_inputs; j++) fused.inputs[j] = mm_node->inputs[j];
            fused.inputs[fused.n_inputs - 1] = bias_id;
        }
        fused.ndim = node->ndim;
        if (node->ndim > 0 && node->shape) {
            fused.shape = (int *)malloc(sizeof(int) * node->ndim);
            if (fused.shape) memcpy(fused.shape, node->shape, sizeof(int) * node->ndim);
        }
        fused.dtype = node->dtype;
        fused.device = node->device;
        plan_add_node(out, &fused);
        if (add_id >= 0 && add_id < n) skip[add_id] = 1;
        if (mm_id >= 0 && mm_id < n) skip[mm_id] = 1;
    }

    free(by_id);
    free(users);
    free(skip);
    graph_plan_free(p);
    return out;
}

static int is_output_id(GraphPlan *p, int id) {
    for (int i = 0; i < p->n_outputs; i++) {
        if (p->outputs[i] == id) return 1;
    }
    return 0;
}

static GraphPlan *fuse_elementwise(GraphPlan *p) {
    int n = p->n_nodes;
    GraphPlan *out = (GraphPlan *)xcalloc(1, sizeof(*out));
    if (!out) return p;
    out->inputs = NULL;
    out->outputs = NULL;
    if (p->n_inputs) {
        out->inputs = (int *)malloc(sizeof(int) * p->n_inputs);
        if (out->inputs) {
            memcpy(out->inputs, p->inputs, sizeof(int) * p->n_inputs);
            out->n_inputs = p->n_inputs;
        }
    }
    if (p->n_outputs) {
        out->outputs = (int *)malloc(sizeof(int) * p->n_outputs);
        if (out->outputs) {
            memcpy(out->outputs, p->outputs, sizeof(int) * p->n_outputs);
            out->n_outputs = p->n_outputs;
        }
    }

    GraphPlanNode **by_id = (GraphPlanNode **)xcalloc(n, sizeof(*by_id));
    int *users = (int *)xcalloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        GraphPlanNode *node = &p->nodes[i];
        if (node->id >= 0 && node->id < n) by_id[node->id] = node;
        for (int j = 0; j < node->n_inputs; j++) {
            int id = node->inputs[j];
            if (id >= 0 && id < n) users[id] += 1;
        }
    }

    int *skip = (int *)xcalloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        GraphPlanNode *node = &p->nodes[i];
        if (node->id >= 0 && node->id < n && skip[node->id]) continue;
        if (!is_elementwise(node->op) || node->n_inputs == 0) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        GraphPlanNode *chain_nodes[64];
        int chain_len = 0;
        GraphPlanNode *curr = node;
        while (curr && is_elementwise(curr->op)) {
            if (chain_len >= 64) break;
            chain_nodes[chain_len++] = curr;
            if (is_output_id(p, curr->id)) break;
            if (users[curr->id] != 1) break;
            GraphPlanNode *next = NULL;
            for (int j = 0; j < n; j++) {
                if (p->nodes[j].n_inputs == 0) continue;
                for (int k = 0; k < p->nodes[j].n_inputs; k++) {
                    if (p->nodes[j].inputs[k] == curr->id) {
                        if (next) { next = NULL; j = n; break; }
                        next = &p->nodes[j];
                    }
                }
            }
            if (!next || !is_elementwise(next->op)) break;
            curr = next;
        }
        if (chain_len <= 1) {
            GraphPlanNode copy = plan_node_clone(node);
            plan_add_node(out, &copy);
            continue;
        }
        for (int c = 1; c < chain_len; c++) {
            int cid = chain_nodes[c]->id;
            if (cid >= 0 && cid < n) skip[cid] = 1;
        }
        GraphPlanNode fused;
        memset(&fused, 0, sizeof(fused));
        fused.op = GRAPH_OP_FUSED_ELEMENTWISE;
        fused.id = chain_nodes[chain_len - 1]->id;
        fused.n_chain = chain_len;
        fused.chain = (GraphNode *)malloc(sizeof(GraphNode) * chain_len);
        if (fused.chain) {
            for (int c = 0; c < chain_len; c++) {
                GraphPlanNode *src = chain_nodes[c];
                GraphNode dst;
                memset(&dst, 0, sizeof(dst));
                dst.id = src->id;
                dst.op = src->op;
                dst.n_inputs = src->n_inputs;
                if (src->n_inputs > 0) {
                    dst.inputs = (int *)malloc(sizeof(int) * src->n_inputs);
                    if (dst.inputs) memcpy(dst.inputs, src->inputs, sizeof(int) * src->n_inputs);
                }
                dst.ndim = src->ndim;
                if (src->ndim > 0 && src->shape) {
                    dst.shape = (int *)malloc(sizeof(int) * src->ndim);
                    if (dst.shape) memcpy(dst.shape, src->shape, sizeof(int) * src->ndim);
                }
                dst.dtype = src->dtype;
                dst.device = src->device;
                fused.chain[c] = dst;
            }
        }
        fused.ndim = chain_nodes[chain_len - 1]->ndim;
        if (chain_nodes[chain_len - 1]->ndim > 0 && chain_nodes[chain_len - 1]->shape) {
            fused.shape = (int *)malloc(sizeof(int) * fused.ndim);
            if (fused.shape) memcpy(fused.shape, chain_nodes[chain_len - 1]->shape, sizeof(int) * fused.ndim);
        }
        fused.dtype = chain_nodes[chain_len - 1]->dtype;
        fused.device = chain_nodes[chain_len - 1]->device;
        plan_add_node(out, &fused);
    }

    free(by_id);
    free(users);
    free(skip);
    graph_plan_free(p);
    return out;
}

GraphPlan *graph_compile(Graph *g) {
    if (!g) return NULL;
    GraphPlan *p = plan_new_from_graph(g);
    if (!p) return NULL;
    for (int i = 0; i < g->n_nodes; i++) {
        GraphPlanNode pn = plan_node_from_graph(&g->nodes[i]);
        plan_add_node(p, &pn);
    }
    p = fuse_matmul_bias_relu(p);
    p = fuse_elementwise(p);
    return p;
}

static Tensor *apply_op(GraphOp op, Tensor **inputs, int n_inputs) {
    if (op == GRAPH_OP_ADD && n_inputs == 2) return tensor_add(inputs[0], inputs[1]);
    if (op == GRAPH_OP_SUB && n_inputs == 2) return tensor_sub(inputs[0], inputs[1]);
    if (op == GRAPH_OP_MUL && n_inputs == 2) return tensor_mul(inputs[0], inputs[1]);
    if (op == GRAPH_OP_DIV && n_inputs == 2) return tensor_div(inputs[0], inputs[1]);
    if (op == GRAPH_OP_MATMUL && n_inputs == 2) return tensor_matmul(inputs[0], inputs[1]);
    if (op == GRAPH_OP_RELU && n_inputs == 1) return tensor_relu(inputs[0]);
    return NULL;
}

Tensor *graph_run(GraphPlan *p, Tensor **inputs, int n_inputs) {
    if (!p) return NULL;
    if (p->n_inputs != n_inputs) return NULL;
    int max_id = 0;
    for (int i = 0; i < p->n_nodes; i++) {
        if (p->nodes[i].id > max_id) max_id = p->nodes[i].id;
    }
    for (int i = 0; i < p->n_outputs; i++) {
        if (p->outputs[i] > max_id) max_id = p->outputs[i];
    }
    Tensor **values = (Tensor **)xcalloc((size_t)max_id + 1, sizeof(*values));
    if (!values) return NULL;
    for (int i = 0; i < p->n_inputs; i++) {
        int id = p->inputs[i];
        if (id >= 0 && id <= max_id) values[id] = inputs[i];
    }

    graph_capture_push_disabled();
    for (int i = 0; i < p->n_nodes; i++) {
        GraphPlanNode *node = &p->nodes[i];
        if (node->op == GRAPH_OP_INPUT) continue;
        if (node->op == GRAPH_OP_FUSED_ELEMENTWISE) {
            for (int c = 0; c < node->n_chain; c++) {
                GraphNode *sub = &node->chain[c];
                Tensor *args[2] = {0};
                for (int j = 0; j < sub->n_inputs; j++) {
                    int id = sub->inputs[j];
                    args[j] = (id >= 0 && id <= max_id) ? values[id] : NULL;
                }
                Tensor *out = apply_op(sub->op, args, sub->n_inputs);
                if (out) values[sub->id] = out;
            }
            continue;
        }
        if (node->op == GRAPH_OP_MATMUL_BIAS_RELU) {
            Tensor *a = values[node->inputs[0]];
            Tensor *b = values[node->inputs[1]];
            Tensor *bias = values[node->inputs[2]];
            Tensor *mm = tensor_matmul(a, b);
            Tensor *sum = tensor_add(mm, bias);
            Tensor *out = tensor_relu(sum);
            values[node->id] = out;
            continue;
        }
        Tensor *args[2] = {0};
        for (int j = 0; j < node->n_inputs; j++) {
            int id = node->inputs[j];
            args[j] = (id >= 0 && id <= max_id) ? values[id] : NULL;
        }
        Tensor *out = apply_op(node->op, args, node->n_inputs);
        if (out) values[node->id] = out;
    }
    graph_capture_pop_disabled();

    Tensor *ret = NULL;
    if (p->n_outputs > 0) {
        int id = p->outputs[0];
        if (id >= 0 && id <= max_id) ret = values[id];
    }
    free(values);
    return ret;
}

Tensor *graph_cache_run(Graph *g, Tensor **inputs, int n_inputs, int *hit) {
    if (!g) return NULL;
    uint64_t key = graph_signature(g);
    GraphPlan *plan = graph_cache_get(key);
    if (plan) {
        if (hit) *hit = 1;
        return graph_run(plan, inputs, n_inputs);
    }
    if (hit) *hit = 0;
    plan = graph_compile(g);
    if (!plan) return NULL;
    graph_cache_put(key, plan);
    return graph_run(plan, inputs, n_inputs);
}

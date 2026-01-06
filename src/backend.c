#include "backend.h"
#include <string.h>
#include <stdlib.h>

static Backend cpu_backend = {
    .name = "cpu",
    .matmul = NULL,
    .conv2d = NULL,
    .add = NULL,
    .mul = NULL,
};

static Backend *current_backend = NULL;
static Backend *registry[8];
static int registry_count = 0;

void backend_set(Backend *b) { current_backend = b; }

Backend *backend_get(void) {
    return current_backend ? current_backend : &cpu_backend;
}

const char *backend_name(void) {
    Backend *b = backend_get();
    return (b && b->name) ? b->name : "cpu";
}

int backend_register(Backend *b) {
    if (!b || !b->name) return 0;
    if (registry_count >= 8) return 0;
    registry[registry_count++] = b;
    return 1;
}

int backend_set_by_name(const char *name) {
    if (!name) return 0;
    if (strcmp(name, "cpu") == 0) { backend_set(NULL); return 1; }
    for (int i = 0; i < registry_count; i++) {
        if (registry[i] && registry[i]->name && strcmp(registry[i]->name, name) == 0) {
            backend_set(registry[i]);
            return 1;
        }
    }
    return 0;
}

__attribute__((constructor))
static void backend_init(void) {
    backend_register(&cpu_backend);
    const char *env = getenv("TINYFIN_BACKEND");
    if (env) {
        backend_set_by_name(env);
    }
}

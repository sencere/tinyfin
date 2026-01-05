#include "scratch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void *arena = NULL;
static size_t arena_size = 0;
static size_t arena_offset = 0;

int scratch_is_from_arena(const void *p) {
    if (!arena || !p) return 0;
    const char *base = (const char *)arena;
    return (p >= (const void *)base) && (p < (const void *)(base + arena_size));
}

static size_t parse_env_size(void) {
    const char *env = getenv("TINYFIN_SCRATCH_BYTES");
    if (!env) return 0;
    return (size_t)strtoull(env, NULL, 10);
}

void scratch_init(size_t bytes) {
    if (arena) return;
    size_t sz = bytes ? bytes : parse_env_size();
    if (sz == 0) sz = 1 << 20; /* default 1MB */
    arena = malloc(sz);
    if (arena) {
        arena_size = sz;
        arena_offset = 0;
    }
}

void *scratch_alloc(size_t bytes) {
    if (arena && arena_offset + bytes <= arena_size) {
        void *p = (char*)arena + arena_offset;
        arena_offset += bytes;
        return p;
    }
    /* fallback to malloc */
    void *p = malloc(bytes);
    return p;
}

void scratch_reset(void) {
    arena_offset = 0;
}

void scratch_shutdown(void) {
    if (arena) free(arena);
    arena = NULL;
    arena_size = 0;
    arena_offset = 0;
}

__attribute__((constructor))
static void scratch_ctor(void) {
    scratch_init(0);
}

#ifndef TINYFIN_SCRATCH_H
#define TINYFIN_SCRATCH_H

#include <stddef.h>

/* Initialize scratch arena; size is optional (0 => default). Not thread-safe. */
void scratch_init(size_t bytes);

/* Obtain temporary storage; falls back to malloc if arena insufficient. */
void *scratch_alloc(size_t bytes);

/* Helper to check if a pointer came from the arena (vs malloc fallback). */
int scratch_is_from_arena(const void *p);

/* Release arena-backed storage and free any malloc fallbacks.
 * Call after each op or forward pass to reuse the arena. */
void scratch_reset(void);

/* Destroy arena (call at shutdown if desired). */
void scratch_shutdown(void);

#endif

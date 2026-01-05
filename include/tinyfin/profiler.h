#ifndef TINYFIN_PROFILER_H
#define TINYFIN_PROFILER_H

#include <stdint.h>

/* Simple lightweight profiler API */
void profiler_begin_op(const char *name);
void profiler_end_op(void);

/* Return a malloc'd string summary; caller should free() it if desired. */
char *profiler_get_summary(void);

#endif

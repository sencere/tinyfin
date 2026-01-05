#define _POSIX_C_SOURCE 199309L
#include "profiler.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#define MAX_RECORDS 4096
#define MAX_NAME_LEN 64

typedef struct {
    char name[MAX_NAME_LEN];
    uint64_t dur_ns;
} ProfRecord;

static ProfRecord records[MAX_RECORDS];
static int rec_pos = 0;
static struct timespec tstart;
static char cur_name[MAX_NAME_LEN];

/* Mark records array initially zeroed */
static int profiler_initialized = 0;

static void ensure_init(void) {
    if (profiler_initialized) return;
    for (int i = 0; i < MAX_RECORDS; i++) { records[i].name[0] = '\0'; records[i].dur_ns = 0; }
    profiler_initialized = 1;
}

void profiler_begin_op(const char *name) {
    if (!name) return;
    ensure_init();
        snprintf(cur_name, MAX_NAME_LEN, "%s", name);
    clock_gettime(CLOCK_MONOTONIC, &tstart);
}

void profiler_end_op(void) {
    struct timespec tend;
    clock_gettime(CLOCK_MONOTONIC, &tend);
    uint64_t start_ns = (uint64_t)tstart.tv_sec * 1000000000ULL + (uint64_t)tstart.tv_nsec;
    uint64_t end_ns = (uint64_t)tend.tv_sec * 1000000000ULL + (uint64_t)tend.tv_nsec;
    uint64_t dur = (end_ns - start_ns);
    int p = rec_pos++;
    if (p >= MAX_RECORDS) p = (rec_pos = 0);
        snprintf(records[p].name, MAX_NAME_LEN, "%s", cur_name);
    records[p].dur_ns = dur;
}

char *profiler_get_summary(void) {
    /* Aggregate by name */
    typedef struct { char name[MAX_NAME_LEN]; uint64_t total; int count; } Agg;
    Agg aggs[256]; int nagg = 0;
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (records[i].dur_ns == 0) continue;
        int found = -1;
        for (int j = 0; j < nagg; j++) if (strncmp(aggs[j].name, records[i].name, MAX_NAME_LEN)==0) { found = j; break; }
        if (found==-1) {
            if (nagg < 256) {
                    size_t _len = strlen(records[i].name);
                    if (_len >= MAX_NAME_LEN) _len = MAX_NAME_LEN-1;
                    memcpy(aggs[nagg].name, records[i].name, _len);
                    aggs[nagg].name[_len] = '\0';
                aggs[nagg].total = records[i].dur_ns;
                aggs[nagg].count = 1;
                nagg++;
            }
        } else {
            aggs[found].total += records[i].dur_ns;
            aggs[found].count += 1;
        }
    }

    /* Build summary string */
    char *out = malloc(4096);
    if (!out) return NULL;
    size_t off = 0;
    off += snprintf(out+off, 4096-off, "Profiler summary:\n");
    for (int i = 0; i < nagg; i++) {
        double ms = (double)aggs[i].total / 1e6;
        off += snprintf(out+off, 4096-off, "%s: total=%.3f ms count=%d avg=%.3f ms\n", aggs[i].name, ms, aggs[i].count, ms / aggs[i].count);
    }
    return out;
}

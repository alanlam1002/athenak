// klog: minimal Kokkos Tools kernel logger for locating an asynchronous GPU fault.
// Each rank writes "B <name>" before and "E" after every parallel_for/reduce/scan to
// klog.<pid>.txt (line-buffered, flushed). Run with KOKKOS_TOOLS_LIBS=<this .so> and
// KOKKOS_TOOLS_GLOBALFENCES=1 so every kernel completes before the next begins; the last
// "B" without an "E" in a crashed rank's file is the faulting kernel.
// Build: gcc -O2 -fPIC -shared -o libklog.so klog.c
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

static FILE *fp;
static void open_log(void) {
  if (fp) return;
  char fn[64];
  snprintf(fn, sizeof(fn), "klog.%d.txt", (int)getpid());
  fp = fopen(fn, "w");
}
void kokkosp_init_library(int lvl, uint64_t ver, uint32_t ndev, void *info) { open_log(); }
void kokkosp_finalize_library(void) { if (fp) fclose(fp); fp = NULL; }
static void b(const char *n) { open_log(); if (fp) { fprintf(fp, "B %s\n", n); fflush(fp); } }
static void e(void) { if (fp) { fputs("E\n", fp); fflush(fp); } }
void kokkosp_begin_parallel_for(const char *n, uint32_t d, uint64_t *k) { b(n); }
void kokkosp_end_parallel_for(uint64_t k) { e(); }
void kokkosp_begin_parallel_reduce(const char *n, uint32_t d, uint64_t *k) { b(n); }
void kokkosp_end_parallel_reduce(uint64_t k) { e(); }
void kokkosp_begin_parallel_scan(const char *n, uint32_t d, uint64_t *k) { b(n); }
void kokkosp_end_parallel_scan(uint64_t k) { e(); }

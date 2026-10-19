// redlog: LD_PRELOAD PMPI interposer logging every MPI_Allreduce on chosen ranks, to find
// which global reduction first differs between two runs from the same restart
// (R-078 item 7, run-to-run non-determinism).
//
// Each logged rank writes redlog.<rank>.txt, one line per call:
//   <seq> <caller addr> <datatype size> <count> | in: <hex of up to 8 local inputs> |
//   out: <hex of up to 8 results>
// (inputs are copied before the call, so MPI_IN_PLACE is handled). Values are written as raw
// 64-bit hex (or 32-bit for 4-byte types), so equal lines mean bitwise-equal values. Symbolise
// a caller with  addr2line -f -C -e athena <addr>.
//   REDLOG_RANKS="0,5"  ranks to log (default 0);  REDLOG_MAX=N  stop after N calls (default 2e6)
// Build: gcc -O2 -fPIC -shared -o libredlog.so redlog.c -I<mpich>/include
#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static FILE *fp;
static int inited, on;
static long seq, maxcalls = 2000000;

static void init(void) {
  if (inited) return;
  inited = 1;
  int r = 0;
  PMPI_Comm_rank(MPI_COMM_WORLD, &r);
  const char *ranks = getenv("REDLOG_RANKS");
  if (ranks == NULL) ranks = "0";
  char buf[256];
  strncpy(buf, ranks, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = 0;
  for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) if (atoi(t) == r) on = 1;
  const char *m = getenv("REDLOG_MAX");
  if (m) maxcalls = atol(m);
  if (on) {
    char fn[64];
    snprintf(fn, sizeof(fn), "redlog.%d.txt", r);
    fp = fopen(fn, "w");
  }
}

static void dump(const void *p, int sz, int n) {
  const unsigned char *c = (const unsigned char *)p;
  for (int i = 0; i < n; ++i) {
    if (sz == 8) { uint64_t v; memcpy(&v, c + 8*i, 8); fprintf(fp, " %016llx", (unsigned long long)v); }
    else if (sz == 4) { uint32_t v; memcpy(&v, c + 4*i, 4); fprintf(fp, " %08x", v); }
    else { fprintf(fp, " ?"); }
  }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype dt, MPI_Op op,
                  MPI_Comm comm) {
  init();
  void *caller = __builtin_return_address(0);
  int sz = 0;
  PMPI_Type_size(dt, &sz);
  int n = count < 8 ? count : 8;
  unsigned char in[64];
  int log = on && fp && seq < maxcalls && (sz == 4 || sz == 8) && n > 0;
  if (log) memcpy(in, sendbuf == MPI_IN_PLACE ? recvbuf : sendbuf, (size_t)n * sz);
  int ret = PMPI_Allreduce(sendbuf, recvbuf, count, dt, op, comm);
  if (log) {
    fprintf(fp, "%ld %p %d %d | in:", seq, caller, sz, count);
    dump(in, sz, n);
    fprintf(fp, " | out:");
    dump(recvbuf, sz, n);
    fputc('\n', fp);
  }
  if (on) ++seq;
  return ret;
}

int MPI_Finalize(void) {
  if (fp) { fclose(fp); fp = NULL; }
  return PMPI_Finalize();
}

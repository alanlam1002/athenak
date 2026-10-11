// reqtrack: LD_PRELOAD PMPI interposer that tracks live MPI requests per rank.
//
// Every nonblocking call that creates a request records (handle -> caller address);
// every completion (Wait*/Test* with the handle nulled, Request_free) removes it.
// Every REQTRACK_SEC seconds (default 30), and at MPI_Finalize, each rank appends
//   <wall> live=<n> created=<c> completed=<d>  top call sites: <addr>:<count> ...
// to reqtrack.<rank>.txt in the working directory. Symbolise the addresses with
//   addr2line -f -C -e athena <addr>
// A leak shows as monotonic growth of `live`, attributed to its creating call site.
//
// Persistent requests (Send_init/Recv_init) stay non-null after completion; they are
// tracked separately (created at *_init, removed at Request_free) and reported as `pers`.
//
// Build: gcc -O2 -fPIC -shared -o libreqtrack.so reqtrack.c -I<mpich>/include -ldl
#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define CAP (1 << 22)                    // open-addressing table (4M slots)
typedef struct { MPI_Request h; void *site; int used; int pers; } slot_t;
static slot_t *tab;
static long live, pers_live, created, completed;
static int rank = -1;
static double t0, tlast, every = 30.0;

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static uint64_t hsh(MPI_Request h) { uint64_t x = (uint64_t)(uintptr_t)h; x *= 0x9E3779B97F4A7C15ull; return x >> 40; }

static void init_tab(void) {
  if (tab) return;
  tab = calloc(CAP, sizeof(slot_t));
  const char *e = getenv("REQTRACK_SEC");
  if (e) every = atof(e);
  t0 = tlast = now();
}

static void add(MPI_Request h, void *site, int pers) {
  if (h == MPI_REQUEST_NULL) return;
  init_tab();
  uint64_t i = hsh(h) & (CAP - 1);
  while (tab[i].used == 1 && tab[i].h != h) i = (i + 1) & (CAP - 1);
  if (tab[i].used == 1) return;          // already tracked (handle reuse after a miss)
  tab[i].h = h; tab[i].site = site; tab[i].used = 1; tab[i].pers = pers;
  if (pers) pers_live++; else live++;
  created++;
}

static void del(MPI_Request h) {
  if (!tab || h == MPI_REQUEST_NULL) return;
  uint64_t i = hsh(h) & (CAP - 1);
  while (tab[i].used) {
    if (tab[i].used == 1 && tab[i].h == h) {
      tab[i].used = 2;                   // tombstone
      if (tab[i].pers) pers_live--; else live--;
      completed++;
      return;
    }
    i = (i + 1) & (CAP - 1);
  }
}

static void report(int force) {
  if (!tab) return;
  double t = now();
  if (!force && t - tlast < every) return;
  tlast = t;
  if (rank < 0) PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // tally live requests by site (small linear table; sites are few)
  enum { NS = 64 };
  void *s[NS]; long c[NS]; int ns = 0;
  for (long i = 0; i < CAP; ++i) {
    if (tab[i].used != 1 || tab[i].pers) continue;
    int k = 0;
    while (k < ns && s[k] != tab[i].site) ++k;
    if (k == ns) { if (ns == NS) continue; s[ns] = tab[i].site; c[ns++] = 0; }
    c[k]++;
  }
  char fn[64]; snprintf(fn, sizeof fn, "reqtrack.%d.txt", rank);
  FILE *f = fopen(fn, "a");
  if (!f) return;
  fprintf(f, "%.1f live=%ld pers=%ld created=%ld completed=%ld sites:", t - t0, live, pers_live,
          created, completed);
  for (int r = 0; r < 8 && r < ns; ++r) {           // top 8 by count
    int b = r;
    for (int k = r + 1; k < ns; ++k) if (c[k] > c[b]) b = k;
    void *ts = s[r]; long tc = c[r]; s[r] = s[b]; c[r] = c[b]; s[b] = ts; c[b] = tc;
    fprintf(f, " %p:%ld", s[r], c[r]);
  }
  fprintf(f, "\n");
  fclose(f);
}

#define SITE __builtin_return_address(0)
#define CREATE(call, req, pers) do { int rc_ = call; add(*(req), SITE, pers); report(0); return rc_; } while (0)

int MPI_Isend(const void *b, int n, MPI_Datatype d, int dst, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Isend(b, n, d, dst, tag, c, r), r, 0); }
int MPI_Issend(const void *b, int n, MPI_Datatype d, int dst, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Issend(b, n, d, dst, tag, c, r), r, 0); }
int MPI_Irsend(const void *b, int n, MPI_Datatype d, int dst, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Irsend(b, n, d, dst, tag, c, r), r, 0); }
int MPI_Ibsend(const void *b, int n, MPI_Datatype d, int dst, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Ibsend(b, n, d, dst, tag, c, r), r, 0); }
int MPI_Irecv(void *b, int n, MPI_Datatype d, int src, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Irecv(b, n, d, src, tag, c, r), r, 0); }
int MPI_Ibarrier(MPI_Comm c, MPI_Request *r) { CREATE(PMPI_Ibarrier(c, r), r, 0); }
int MPI_Iallreduce(const void *sb, void *rb, int n, MPI_Datatype d, MPI_Op op, MPI_Comm c,
                   MPI_Request *r) { CREATE(PMPI_Iallreduce(sb, rb, n, d, op, c, r), r, 0); }
int MPI_Ibcast(void *b, int n, MPI_Datatype d, int root, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Ibcast(b, n, d, root, c, r), r, 0); }
int MPI_Send_init(const void *b, int n, MPI_Datatype d, int dst, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Send_init(b, n, d, dst, tag, c, r), r, 1); }
int MPI_Recv_init(void *b, int n, MPI_Datatype d, int src, int tag, MPI_Comm c, MPI_Request *r)
{ CREATE(PMPI_Recv_init(b, n, d, src, tag, c, r), r, 1); }

int MPI_Wait(MPI_Request *r, MPI_Status *st) {
  MPI_Request h = *r; int rc = PMPI_Wait(r, st);
  if (*r == MPI_REQUEST_NULL) del(h);
  report(0); return rc;
}
int MPI_Test(MPI_Request *r, int *flag, MPI_Status *st) {
  MPI_Request h = *r; int rc = PMPI_Test(r, flag, st);
  if (*flag && *r == MPI_REQUEST_NULL) del(h);
  report(0); return rc;
}
int MPI_Waitall(int n, MPI_Request r[], MPI_Status st[]) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Waitall(n, r, st);
  for (int i = 0; i < n; ++i) if (r[i] == MPI_REQUEST_NULL) del(h[i]);
  free(h); report(0); return rc;
}
int MPI_Testall(int n, MPI_Request r[], int *flag, MPI_Status st[]) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Testall(n, r, flag, st);
  if (*flag) for (int i = 0; i < n; ++i) if (r[i] == MPI_REQUEST_NULL) del(h[i]);
  free(h); report(0); return rc;
}
int MPI_Waitany(int n, MPI_Request r[], int *idx, MPI_Status *st) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Waitany(n, r, idx, st);
  if (*idx != MPI_UNDEFINED && r[*idx] == MPI_REQUEST_NULL) del(h[*idx]);
  free(h); report(0); return rc;
}
int MPI_Testany(int n, MPI_Request r[], int *idx, int *flag, MPI_Status *st) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Testany(n, r, idx, flag, st);
  if (*flag && *idx != MPI_UNDEFINED && r[*idx] == MPI_REQUEST_NULL) del(h[*idx]);
  free(h); report(0); return rc;
}
int MPI_Waitsome(int n, MPI_Request r[], int *oc, int idx[], MPI_Status st[]) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Waitsome(n, r, oc, idx, st);
  if (*oc != MPI_UNDEFINED) for (int i = 0; i < *oc; ++i) if (r[idx[i]] == MPI_REQUEST_NULL) del(h[idx[i]]);
  free(h); report(0); return rc;
}
int MPI_Testsome(int n, MPI_Request r[], int *oc, int idx[], MPI_Status st[]) {
  MPI_Request *h = malloc(sizeof(MPI_Request) * (n > 0 ? n : 1));
  memcpy(h, r, sizeof(MPI_Request) * n);
  int rc = PMPI_Testsome(n, r, oc, idx, st);
  if (*oc != MPI_UNDEFINED) for (int i = 0; i < *oc; ++i) if (r[idx[i]] == MPI_REQUEST_NULL) del(h[idx[i]]);
  free(h); report(0); return rc;
}
int MPI_Request_free(MPI_Request *r) { MPI_Request h = *r; int rc = PMPI_Request_free(r); del(h); return rc; }
int MPI_Finalize(void) { report(1); return PMPI_Finalize(); }

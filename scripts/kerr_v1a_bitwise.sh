#!/bin/bash
# V1a (K-001): the a = 0 / option-off bitwise gate of wip/kerr against proj/tde.
# Runs the B.1 deck (or any deck) for 40 cycles with per-rank checksums of cycles 38-39
# (<job> checksum_mg, as the main session's G1 gates, c203_gate/t203g.sh) under two binaries
# and compares chksum.*.txt and the .hst files byte for byte. Rerun after every step and
# every rebase of wip/kerr.
#
#   kerr_v1a_bitwise.sh DECK REF_BIN NEW_BIN OUTDIR [NODEFILE] [EXTRA_KEY=VAL ...]
#
# Environment: the caller sets up the MPI/GPU environment (tde_mpi_env.sh, aurora_guard.sh).
#   NP (12), PPN (12), CPU_BIND, WRAP (gpu_tile_compact.sh), NLIM (40), TIMEOUT (2400 s).
#   ARMS (both | ref | new): which arms to run; the other is reused from OUTDIR (one arm per
#     job). REQUIRE (state | strict): the verdict needed for exit code 0.
# With two or more hosts in NODEFILE and ARMS=both the two binaries run concurrently, one host each.
# The rule (user-approved, RK-004 §8.2): STATE after every step and rebase; before any merge into
# proj/tde, one STRICT pass with the arms in separate jobs:
#   job 1: ARMS=ref kerr_v1a_bitwise.sh ...;  job 2: ARMS=new REQUIRE=strict kerr_v1a_bitwise.sh ...
# Exit code 0 = the required verdict, 1 = not, 2 = setup error.
set -u
DECK=$(readlink -f "$1"); REF=$(readlink -f "$2"); NEW=$(readlink -f "$3"); OUT=$4
NODEFILE=${5:-${PBS_NODEFILE:-}}; shift 5 2>/dev/null || shift $#
NP=${NP:-12}; PPN=${PPN:-12}; NLIM=${NLIM:-40}
CPU_BIND=${CPU_BIND:-"list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"}
WRAP=${WRAP:-/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh}
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
# the gate deck: checksums, nlim, and only the history output (no restarts, no slices)
python3 - "$DECK" "$OUT/gate.par" "$NLIM" "$@" <<'EOF'
import re, sys
src, dst, nlim, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
out, blk, skip, pend = [], None, False, None
for line in open(src):
    m = re.match(r'^<([^>]+)>', line)
    if m:
        blk = m.group(1)
        skip = False
        if blk.startswith('output'):
            skip = True          # decided at file_type below
            pend = [line]
            continue
        out.append(line)
        if blk == 'job':
            out += ['checksum_mg = 1      # V1a gate (kerr_v1a_bitwise.sh)\n',
                    'checksum_cycles = 2\n', 'checksum_start = %d\n' % (int(nlim) - 2)]
        continue
    if blk and blk.startswith('output'):
        if skip and pend is not None:
            pend.append(line)
            if re.match(r'^\s*file_type\s*=\s*hst\b', line):
                out += pend; pend = None; skip = False
            continue
        if not skip:
            out.append(line)
        continue
    if blk == 'time' and re.match(r'^\s*nlim\s*=', line):
        line = 'nlim = %s   # V1a gate\n' % nlim
    out.append(line)
for kv in extra:              # block/key=value
    b, kv2 = kv.split('/', 1); k, v = kv2.split('=', 1)
    txt = ''.join(out)
    pat = re.compile(r'(^<%s>\s*\n)' % re.escape(b), re.M)
    if not pat.search(txt):
        sys.exit('no block <%s> in the deck' % b)
    out = [pat.sub(r'\g<1>%s = %s\n' % (k, v), txt, count=1)]
open(dst, 'w').write(''.join(out))
EOF
[ $? -eq 0 ] || exit 2
mapfile -t HOSTS < <(sort -u "$NODEFILE")
run() {  # tag binary host
  local tag=$1 bin=$2 host=$3
  rm -rf "$OUT/$tag"; mkdir -p "$OUT/$tag"; cd "$OUT/$tag"
  echo "$tag start $(date -u +%FT%TZ) host $host bin $(sha256sum "$bin" | cut -c1-8)" >> "$OUT/runs.log"
  timeout "${TIMEOUT:-2400}" mpiexec -np "$NP" -ppn "$PPN" --hosts "$host" --cpu-bind="$CPU_BIND" \
    "$WRAP" "$bin" -d "$OUT/$tag" -i "$OUT/gate.par" > out.log 2> err.log
  echo "$tag rc=$? $(grep -o 'cycle=[0-9]* time=[0-9.e+-]*' out.log | tail -1) end $(date -u +%FT%TZ)" >> "$OUT/runs.log"
}
ARMS=${ARMS:-both}; REQUIRE=${REQUIRE:-state}
case "$ARMS" in
  both) if [ ${#HOSTS[@]} -ge 2 ]; then
          ( run ref "$REF" "${HOSTS[0]}" ) & ( run new "$NEW" "${HOSTS[1]}" ) & wait
        else
          run ref "$REF" "${HOSTS[0]}"; run new "$NEW" "${HOSTS[0]}"
        fi ;;
  ref)  run ref "$REF" "${HOSTS[0]}"
        echo "V1a: ref arm only (ARMS=ref); run ARMS=new on the same OUTDIR to compare" | tee -a "$OUT/runs.log"
        exit 0 ;;
  new)  [ -d "$OUT/ref" ] || { echo "ARMS=new: no $OUT/ref (run ARMS=ref first)"; exit 2; }
        run new "$NEW" "${HOSTS[0]}" ;;
  *)    echo "ARMS must be both, ref or new"; exit 2 ;;
esac
echo "arms: $ARMS, required verdict: $REQUIRE" >> "$OUT/runs.log"
# compare: STRICT = every checksum and .hst file byte-identical. Otherwise STATE = .hst files
# identical, the same checksum keys (cycle stage task occurrence array) on every rank, every
# end-of-stage row identical, and every differing row a cross-list snapshot (an MHD_* task
# row of a CFC-side array, or a CFC_* task row of a hydro array): the transient
# task-interleaving rows of C-090 (code/OUTBOX.md), which re-converge. Anything else: FAIL.
cd "$OUT"
for f in $(cd ref && ls *.hst 2>/dev/null); do
  cmp -s "ref/$f" "new/$f" || echo "DIFF $f"
done > hst.diff
python3 - "$OUT" "$(cat hst.diff | wc -l)" "$(cd ref && ls *.hst 2>/dev/null | wc -l)" "$REF" "$NEW" "$REQUIRE" <<'PYEOF' | tee -a "$OUT/runs.log"
import collections, filecmp, glob, hashlib, os, sys
out, nhdiff, nh, ref, new, require = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
                                      sys.argv[5], sys.argv[6])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
hydro = {'u0', 'u1', 'w0', 'flx1', 'flx2', 'flx3', 'fofc', 'temp', 'blk_u0', 'blk_w0', 'blk_temp'}
files = sorted(os.path.basename(f) for f in glob.glob(out + '/ref/chksum.*.txt'))
nnew = len(glob.glob(out + '/new/chksum.*.txt'))
strict = files and len(files) == nnew and nhdiff == 0 and nh > 0 and all(
    filecmp.cmp(f'{out}/ref/{f}', f'{out}/new/{f}', shallow=False) for f in files)
bad, trans, endrows = [], 0, 0
if not files or len(files) != nnew: bad.append(f'checksum files ref {len(files)} new {nnew}')
if nh == 0 or nhdiff: bad.append(f'hst: {nh} files, {nhdiff} differ')
for f in files:
    A, B, order = {}, {}, []
    for d, D in (('ref', A), ('new', B)):
        for l in open(f'{out}/{d}/{f}'):
            x = l.split()
            if len(x) < 6: continue
            k = tuple(x[:5]); D[k] = x[5:]
            if d == 'ref': order.append(k)
    if set(A) != set(B):
        bad.append(f'{f}: key sets differ ({len(set(A) ^ set(B))})'); continue
    last = {}
    for k in order: last[k[:2]] = k[:4]
    for k in order:
        if A[k] == B[k]: continue
        if last[k[:2]] == k[:4]:
            bad.append(f'{f}: end-of-stage row differs {k}'); continue
        task, arr = k[2], k[4]
        if (task.startswith('MHD_') and arr not in hydro) or (task.startswith('CFC_') and arr in hydro):
            trans += 1
        else:
            bad.append(f'{f}: state row differs {k}')
    endrows += sum(1 for k in order if last[k[:2]] == k[:4])
for b in bad[:20]: print('  ' + b)
verdict = 'PASS (STRICT: all files byte-identical)' if strict else (
    f'PASS (STATE: {endrows} end-of-stage rows and {nh} hst files identical; {trans} transient '
    'task-interleaving rows, C-090)' if not bad else f'FAIL ({len(bad)} problems)')
ok = strict if require == 'strict' else (strict or not bad)
print(f'V1a {verdict if ok or bad else verdict.replace("PASS", "FAIL (STRICT required)")}: '
      f'{len(files)} checksum files; ref {sha(ref)} new {sha(new)}')
sys.exit(0 if ok else 1)
PYEOF
exit ${PIPESTATUS[0]}

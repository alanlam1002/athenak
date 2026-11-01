#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=21
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c160_tsk/td160g.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c160_tsk/td160g.err
# c160 (R-194 items 1, 2, 3, 5, user-approved 2026-10-31): gates + ISO_A, debug-scaling, 21 nodes x 1 h; binary athena_tsk =
# wip/tau-source-k (<dyn_grmhd> tau_source_K = keff | kcfc | ktr; <mhd> killing_full_rest_mass = explicit):
#   gKv*/gKe*/gKc*/gKt* + GATE_flat   flat static metric, beta' = 0: keff, kcfc, ktr == Valencia bitwise (1 and 20 cycles)
#   Pm + GATE_pm                      ghost/owner probe of the T face flux, rest mass explicit (KF_i's deck as arm ii, 2 cycles)
#   IKe, IKc, IKt                     ISO_A (Valencia ISO_A deck, L4, v = 0.4 comoving, self-gravity, DE off) t = 0 -> 120, after GATE_flat
#   V0o/V0n, K0o/K0n + GATE_d0        default bitwise on GPU: 226acf2b vs the new binary, Valencia and arm (ii), 15 B.1 cycles
#   SZ80/SZ116/SZ120                  R-194 item 6: one cycle from B1Z's restarts on B1Z's binary 226acf2b (t = 80.0, 116.1, 120.0) with 3D adm + prim dumps
#   R?1, PICK_R?2, R?2, GATE_r?       restart bitwise across stages and regrids (cycle 515 -> 530): keff (e), kcfc (c), ktr (t),
#                                     rest mass explicit (m)
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c160_tsk
X0=$E/athena_tsk
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X0
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $E/nodes.$PBS_JOBID
T0=$(date +%s)
slot() {   # first-node-line nnodes arms...
  local n0=$1 nn=$2; shift 2
  for a in "$@"; do
    local d=$E/$a r left
    case $a in GATE_*)
      python3 $E/gate_${a#GATE_}.py $E >> $E/runs.log 2>&1 || { echo "$a failed: slot stops $(date -u +%FT%TZ)" >> $E/runs.log; touch $E/FAILED_$a.$PBS_JOBID; return; }
      touch $E/PASS_$a.$PBS_JOBID; echo "$a passed $(date -u +%FT%TZ)" >> $E/runs.log
      continue ;;
    PICK_*)   # FROM_<arm>: the earliest restart of the arm before it (ZR1's cycle-515 file)
      src=${a#PICK_}; ls $E/${src%2}1/rst/*.rst 2>/dev/null | LC_ALL=C sort | head -1 > $E/FROM_${a#PICK_}
      echo "$a: $(cat $E/FROM_${a#PICK_}) $(date -u +%FT%TZ)" >> $E/runs.log
      continue ;;
    WAIT_*)
      local gw=GATE_${a#WAIT_} wn=0
      while [ ! -e $E/PASS_$gw.$PBS_JOBID ] && [ ! -e $E/FAILED_$gw.$PBS_JOBID ] && [ $wn -lt 150 ]; do sleep 10; wn=$((wn + 1)); done
      [ -e $E/PASS_$gw.$PBS_JOBID ] || { echo "$a: no pass, slot stops $(date -u +%FT%TZ)" >> $E/runs.log; return; }
      continue ;;
    esac
    [ -e $E/DONE_$a ] && continue
    local X=$X0; [ -e $E/BIN_$a ] && X=$(cat $E/BIN_$a)
    left=$(( 3420 - ($(date +%s) - T0) ))
    [ $left -lt 300 ] && { echo "$a deferred (no time left) $PBS_JOBID" >> $E/runs.log; continue; }
    mkdir -p $d && cd $d
    sed -n "${n0},$((n0 + nn - 1))p" $E/nodes.$PBS_JOBID > $E/nodes.$a
    r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
    if [ -z "$r" ] && [ -e $E/FROM_$a ]; then
      local f; f=$(cat $E/FROM_$a)
      case $f in *.rst) r=$f ;; *) r=$(ls $E/$f/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1) ;; esac
      [ -n "$r" ] || { echo "$a: FROM_$a ($f) has no restart, skipped $(date -u +%FT%TZ)" >> $E/runs.log; continue; }
    fi
    echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $X | cut -c1-8) ranks $((12*nn))" >> $E/runs.log
    mpiexec -np $((12*nn)) -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
      $X -d $d -t $(printf '%02d:%02d:00' $(( (left - 120)/3600 )) $(( ((left - 120)%3600)/60 ))) ${r:+-r $r} -i $E/$a.par >> out.log 2>> err.log
    local rc=$?
    echo "$a rc=$rc $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
    case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*|*"cycle limit"*) touch $E/DONE_$a ;; esac
  done
}
( slot 1 1 gKv1 gKe1 gKc1 gKt1 gKv20 gKe20 gKc20 gKt20 GATE_flat ) &
( slot 2 1 Pm GATE_pm ) &
( slot 3 1 WAIT_flat IKe ) &
( slot 4 1 WAIT_flat IKc ) &
( slot 5 1 WAIT_flat IKt ) &
( slot 6 8 V0o V0n K0o K0n GATE_d0 Re1 PICK_Re2 Re2 GATE_re Rc1 PICK_Rc2 Rc2 GATE_rc ) &
( slot 14 8 SZ80 SZ116 SZ120 Rt1 PICK_Rt2 Rt2 GATE_rt Rm1 PICK_Rm2 Rm2 GATE_rm ) &
wait

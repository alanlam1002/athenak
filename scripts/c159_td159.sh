#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=42
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_entry/td159.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_entry/td159.err
# c159 (R-188 items 1 + 5, user-approved 2026-10-31), debug-scaling, 42 nodes x 1 h; binary athena_tb = wip/tau-budget b992fa7c
# (= wip/killing-dtg 6fffe89b + R-188's entry fix and comoving killing_ta + the tau budget), old = c158 athena_tbfix (BIN_<arm>):
#   E1K   killing_full (TB1F's deck) from B1N_cell t = 41.78, 15 cycles, KFSPLIT + tau budget from the first stage, new binary
#   E1K0  the same, the old binary (the entry as in B1K/TB1F/KW: the restart_state block dropped)
#   E1V   Valencia (B1N_cell's deck) + the tau budget (D, D eps per bin), 15 cycles: the entry's reference
#   E1T   comoving killing_t1 (B1T's deck) + tau budget, 15 cycles; E1Ta the same with killing_ta (M_T: the motion source's check)
#   gG1v*/gG1a* + GATE_g1a   G1: no hole (comoving ISO bed), killing_ta vs Valencia, 1 and 20 cycles (1 node, 12 ranks)
#   gG2r/gG2a + GATE_g2a     G2: BH-frame ta vs comoving ta at xidot = 0 (Michel L2, analytic_bg), 100 cycles, bitwise
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c159_entry
X0=$E/athena_tb
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
( slot 1 8 E1K ) &
( slot 9 8 E1K0 ) &
( slot 17 8 E1V ) &
( slot 25 8 E1T ) &
( slot 33 8 E1Ta ) &
( slot 41 1 gG1v1 gG1a1 gG1v20 gG1a20 GATE_g1a ) &
( slot 42 1 gG2r gG2a GATE_g2a ) &
wait

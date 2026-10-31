#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q capacity
#PBS -l select=16
#PBS -l walltime=02:30:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_cap/td159c.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_cap/td159c.err
# c159 (R-188 items 2 + 5, R-190 item 3 and R-191 item 3, user-approved 2026-10-31), capacity, 16 nodes x 2.5 h (one TDE capacity slot), 96 ranks per arm, side by side;
# binary athena_tb = wip/tau-budget c95c6f47; both from B1N_cell's t = 41.78 restart to t = 120, the tau budget (bins star) throughout:
#   VB    Valencia (B1N_cell's deck): the matched tau budget (D, D eps per bin; item 2 reads t = 75-120 against KW)
#   B1Ta  comoving killing_ta (B1T's deck, energy_variable = killing_ta), Delta and M_T per bin (item 5)
#   B1Z   KF_z (R-190: TB1F's killing_full deck with energy_variable = killing_fz), after VB in its slot; Delta_z per bin
#   B1C   KF_CFC (R-191: the same with energy_variable = killing_fcfc), after B1Ta in its slot
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c159_cap
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
    left=$(( 9000 - ($(date +%s) - T0) ))
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
( slot 1 8 VB B1Z ) &
( slot 9 8 B1Ta B1C ) &
wait

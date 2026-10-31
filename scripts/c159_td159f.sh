#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=17
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_fix/td159f.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_fix/td159f.err
# c159 fixes, binary athena_tb 226acf2b = wip/tau-budget 748cac72 (comoving killing_ta: no motion source for D's weight):
#   E1Ta / E1T: 15 cycles of B.1 from B1N_cell t = 41.78 with the tau budget per stage, same binary (Delta_ta vs Delta_t1)
#   gZi / gZz / gCc: the KF source-zero gate (flat static metric: killing_fz and killing_fcfc == killing_full arm (i) bitwise)
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c159_fix; X0=$E/athena_tb
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X0
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
NF=$E/nodes.$PBS_JOBID; sort -u $PBS_NODEFILE > $NF
T0=$(date +%s)
run() {   # arm n0 nn
  local a=$1 n0=$2 nn=$3 d=$E/$1 r left
  [ -e $E/DONE_$a ] && return 0
  left=$(( 1620 - ($(date +%s) - T0) ))
  [ $left -lt 180 ] && { echo "$a deferred (no time left) $PBS_JOBID" >> $E/runs.log; return 1; }
  mkdir -p $d && cd $d
  sed -n "${n0},$((n0 + nn - 1))p" $NF > $E/nodes.$a
  r=""; [ -e $E/FROM_$a ] && r=$(cat $E/FROM_$a)
  echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $X0 | cut -c1-8) ranks $((12*nn))" >> $E/runs.log
  mpiexec -np $((12*nn)) -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
    $X0 -d $d -t $(printf '%02d:%02d:00' $(( (left - 60)/3600 )) $(( ((left - 60)%3600)/60 ))) ${r:+-r $r} -i $E/$a.par >> out.log 2>> err.log
  echo "$a rc=$? $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
  case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*|*"cycle limit"*) touch $E/DONE_$a ;; esac
}
( run E1Ta 1 8 ) &
( run E1T 9 8 ) &
( for a in gZi1 gZz1 gCc1 gZi20 gZz20 gCc20; do run $a 17 1; done
  if python3 $E/gate_si.py $E >> $E/runs.log 2>&1; then echo "GATE_si passed $(date -u +%FT%TZ)" >> $E/runs.log
  else echo "GATE_si failed $(date -u +%FT%TZ)" >> $E/runs.log; fi ) &
wait

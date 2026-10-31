#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=20
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_kfz/td159z.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_kfz/td159z.err
# c159 (R-190 items 1-3 and R-191 items 1-3, user-approved 2026-10-31): KF_z (killing_fz) and KF_CFC (killing_fcfc) gates +
# the ISO_A tests, debug-scaling, 20 nodes x 1 h; binary athena_tb = wip/tau-budget c95c6f47 (KF_z, KF_CFC, R-188's entry fix,
# comoving ta, the tau budget). Each KF_z arm below has its KF_CFC twin (gCc*, CP, IC, E1C, CR1/CR2, gates gc/cp/cr):
#   gZv*/gZz* + GATE_gz  flat gate: ISO bed, residual_scale 0, frame at rest (alpha = 1, beta' = 0): KF_z == Valencia to round-off
#   ZP + GATE_zp         ghost/owner probe of the T face flux (KF_i's ISO_A deck with killing_fz, 2 cycles)
#   IZ                   ISO_A (L4 deck KF_i, v = 0.4 comoving, self-gravity, DE off) with killing_fz, t = 0 -> 120
#   E1Z                  B.1 entry, 15 cycles from B1N_cell t = 41.78, tau budget (cf. c159_entry E1K/E1V)
#   ZR1, PICK_ZR2, ZR2, GATE_zr   restart bitwise across stages and regrids (cycle 515 restart, to 530)
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c159_kfz
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
( slot 1 1 gZv1 gZz1 gZv20 gZz20 GATE_gz gCc1 gCc20 GATE_gc ) &
( slot 2 1 ZP GATE_zp CP GATE_cp ) &
( slot 3 1 IZ ) &
( slot 4 1 IC ) &
( slot 5 8 E1Z ZR1 PICK_ZR2 ZR2 GATE_zr ) &
( slot 13 8 E1C CR1 PICK_CR2 CR2 GATE_cr ) &
wait

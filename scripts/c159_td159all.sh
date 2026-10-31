#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=71
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_entry/td159all.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_entry/td159all.err
# c159 (R-188 items 1, 3, 5; R-190; R-191; user-approved 2026-10-31): the three debug-scaling jobs td159 + td159z + td159k in one,
# debug-scaling 71 nodes x 1 h. Each slot carries its run directory E, default binary X0 (BIN_<arm> overrides) and deck directory:
#   c159_entry (nodes 1-42, athena_tb 6739bc62 = wip/tau-budget b992fa7c; E1K0 = c158 athena_tbfix befb5d84):
#     E1K / E1K0 / E1V / E1T / E1Ta (15 cycles from B1N_cell t = 41.78, tau budget per stage), ta gates G1 (gG1v/gG1a) and G2 (gG2r/gG2a)
#   c159_kfz (nodes 43-62, athena_tb 3f87efb1 = c95c6f47): KF_z and KF_CFC: flat gates gZ/gC, probes ZP/CP, ISO_A IZ/IC (t = 0-120),
#     B.1 entries E1Z/E1C, restart gates ZR/CR (cycle-515 restart to 530)
#   c159_ks (nodes 63-71, athena_ks 20122951 = wip/ks-fixed 35287253): Step B, fixed-KS Michel V/T/A x L2/L3/L4 to t = 120
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
NF=$D/c159_entry/nodes.$PBS_JOBID
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $D/c159_kfz/athena_tb
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $NF
T0=$(date +%s)
slot() {   # rundir default-binary deckdir first-node-line nnodes arms...
  local E=$1 X0=$2 DD=$3 n0=$4 nn=$5; shift 5
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
    sed -n "${n0},$((n0 + nn - 1))p" $NF > $E/nodes.$a
    r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
    if [ -z "$r" ] && [ -e $E/FROM_$a ]; then
      local f; f=$(cat $E/FROM_$a)
      case $f in *.rst) r=$f ;; *) r=$(ls $E/$f/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1) ;; esac
      [ -n "$r" ] || { echo "$a: FROM_$a ($f) has no restart, skipped $(date -u +%FT%TZ)" >> $E/runs.log; continue; }
    fi
    echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $X | cut -c1-8) ranks $((12*nn))" >> $E/runs.log
    mpiexec -np $((12*nn)) -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
      $X -d $d -t $(printf '%02d:%02d:00' $(( (left - 120)/3600 )) $(( ((left - 120)%3600)/60 ))) ${r:+-r $r} -i $DD/$a.par >> out.log 2>> err.log
    local rc=$?
    echo "$a rc=$rc $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
    case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*|*"cycle limit"*) touch $E/DONE_$a ;; esac
  done
}
A=$D/c159_entry; XA=$A/athena_tb
Z=$D/c159_kfz; XZ=$Z/athena_tb
K=$D/c159_ks; XK=$K/athena_ks
( slot $A $XA $A 1 8 E1K ) &
( slot $A $XA $A 9 8 E1K0 ) &
( slot $A $XA $A 17 8 E1V ) &
( slot $A $XA $A 25 8 E1T ) &
( slot $A $XA $A 33 8 E1Ta ) &
( slot $A $XA $A 41 1 gG1v1 gG1a1 gG1v20 gG1a20 GATE_g1a ) &
( slot $A $XA $A 42 1 gG2r gG2a GATE_g2a ) &
( slot $Z $XZ $Z 43 1 gZv1 gZz1 gZv20 gZz20 GATE_gz gCc1 gCc20 GATE_gc ) &
( slot $Z $XZ $Z 44 1 ZP GATE_zp CP GATE_cp ) &
( slot $Z $XZ $Z 45 1 IZ ) &
( slot $Z $XZ $Z 46 1 IC ) &
( slot $Z $XZ $Z 47 8 E1Z ZR1 PICK_ZR2 ZR2 GATE_zr ) &
( slot $Z $XZ $Z 55 8 E1C CR1 PICK_CR2 CR2 GATE_cr ) &
( slot $K $XK $K/decks 63 2 SBV_L4 ) &
( slot $K $XK $K/decks 65 2 SBT_L4 ) &
( slot $K $XK $K/decks 67 2 SBA_L4 ) &
( slot $K $XK $K/decks 69 1 SBV_L3 SBV_L2 ) &
( slot $K $XK $K/decks 70 1 SBT_L3 SBT_L2 ) &
( slot $K $XK $K/decks 71 1 SBA_L3 SBA_L2 ) &
wait

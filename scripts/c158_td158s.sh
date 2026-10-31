#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c158_san/td158s.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c158_san/td158s.err
# c158 (R-186 item 2, user-approved 2026-10-30): the Mm moving-bed deck's post-regrid GPU page fault (KG = c155_b1k/KG.par, 12 ranks):
#   node 1: KGa  wip/killing-dtg 7a5ce960 with device AddressSanitizer (icpx -Xarch_device -fsanitize=address; Intel's memcheck),
#           then KGm  with device MemorySanitizer (-fsanitize=memory; initcheck), each through the cycle-1 regrid (nlim 2)
#   node 2: KGf  wip/testflow-regrid 6fffe89b (the hold rebuilt after a regrid), the full KG deck (nlim 6), then gate_kg.py
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c158_san
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $E/athena_tf
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $E/nodes.$PBS_JOBID
sed -n 1p $E/nodes.$PBS_JOBID > $E/nodes.n1
sed -n 2p $E/nodes.$PBS_JOBID > $E/nodes.n2
run() {   # case binary hostfile walltime [args...]
  local c=$1 x=$2 hf=$3 wt=$4; shift 4
  local d=$E/$c; mkdir -p $d && cd $d
  echo "$c start $(date -u +%FT%TZ) job $PBS_JOBID bin $(sha256sum $x | cut -c1-8) $*" >> $E/runs.log
  timeout $wt mpiexec -np 12 -ppn 12 --hostfile $hf --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
    $x -d $d -i $E/KG.par "$@" > out.log 2> err.log
  echo "$c rc=$? $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) san=$(cat err.log out.log | grep -ci sanitizer) end $(date -u +%FT%TZ)" >> $E/runs.log
}
( run KGa $E/athena_dasan $E/nodes.n1 1500 time/nlim=2; run KGm $E/athena_dmsan $E/nodes.n1 1500 time/nlim=2 ) &
( run KGf $E/athena_tf $E/nodes.n2 2400; mkdir -p $E/gate/KG && ln -sf $E/KGf/out.log $E/gate/KG/out.log
  python3 $E/gate_kg.py $E/gate >> $E/runs.log 2>&1 ) &
wait

#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
# c203 chain (R-239 items 3-5, user-approved 2026-11-06): one self-chaining capacity job holding every TDE science run,
# each on its OWN fixed node count for life (R-239 item 1). Generalises c198's tp198.sh:
#   $E/ARMS lists "arm nodes" (read at every segment start, so arms can be added between segments); deck $E/<arm>.par;
#   STAGES_<arm> lists "t_end label [embedW]" (the first line's t is the segment's time/tlim); a stage done on the time
#   limit -> its latest restart is hard-linked to checkpoints/<label> (+ an optional R1 embed, as tp198.sh); no lines
#   left = STOP_<arm>; any ending without "Terminating" = STOP_<arm>.
#   The successor asks for the sum of the live arms' nodes; it is not submitted when $E/STOP exists or no arm is live.
#   qsub -q capacity -l select=<sum> -l walltime=06:00:00 -v WT=05:50:00 -N c203 -o $E/c203.out -e $E/c203.err tp203.sh
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c203_chain
S=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts
X=$E/athena_c203
live_nodes() { local s=0 a n; while read -r a n; do [ -z "$a" ] || [ "${a:0:1}" = "#" ] || [ -e $E/STOP_$a ] || s=$((s + n)); done < $E/ARMS; echo $s; }
nl=$(live_nodes)
if [ -z "${NOCHAIN:-}" ] && [ ! -e $E/STOP ] && [ $nl -gt 0 ]; then
  N=$(qsub -W depend=afterany:$PBS_JOBID -q $PBS_QUEUE -l select=$nl -l walltime=$(qstat -f $PBS_JOBID | awk '/Resource_List.walltime/{print $3}') -v WT=$WT -N $PBS_JOBNAME -o $E/$PBS_JOBNAME.out -e $E/$PBS_JOBNAME.err $E/tp203.sh) \
    && echo "$(date -u +%FT%TZ) $PBS_JOBID queued successor $N (${nl}n)" >> $E/chain.log \
    && echo "- $(date -u +%FT%TZ) tde c203 $PBS_JOBNAME successor $N $PBS_QUEUE ${nl}n 6h (auto-chain)" >> /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/QUEUE.md
fi
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $E/nodes.$PBS_JOBID
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3
XS=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/xpu-smi-1.2.43-d4w7vpf
stage_done() {  # arm: the stage in STAGES_<arm> line 1 is complete
  local a=$1 d=$E/$1 r t lab w
  read -r t lab w < <(head -1 $E/STAGES_$a)
  r=$(ls $d/rst/*.rst | LC_ALL=C sort | tail -1)
  mkdir -p $d/checkpoints/$lab && ln -f $r $d/checkpoints/$lab/ && cp $d/*.hst $d/parfile.par $d/checkpoints/$lab/ 2>/dev/null
  echo "$(date -u +%FT%TZ) $a stage $lab (t=$t) done: checkpoint $(basename $r)" >> $E/runs.log
  if [ -n "${w:-}" ]; then
    local W=${w#embed} out=${r%.rst}x${w#embed}.rst
    cp -n $E/$a.par $E/$a.par.pre_e$W
    $PY $S/r1_embed.py $r $out --deck $E/$a.par.pre_e$W --deck-out $E/$a.par --manifest ${out%.rst}.json >> $E/embed_$a.log 2>&1 \
      && echo "$(date -u +%FT%TZ) $a R1 embed: $(basename $r) -> $(basename $out) (box width $W)" >> $E/runs.log \
      || { touch $E/STOP_$a; echo "$a R1 embed FAILED (see embed_$a.log): stopped" >> $E/runs.log; return; }
  fi
  sed -i 1d $E/STAGES_$a
  [ -s $E/STAGES_$a ] || { touch $E/STOP_$a; echo "$a: last stage done, stopped" >> $E/runs.log; }
}
arm() {  # name nodes first-node
  local a=$1 nn=$2 d=$E/$1 r ts
  mkdir -p $d && cd $d && cp -n $E/$a.par $d/parfile.par
  sed -n "$3,$(( $3 + nn - 1 ))p" $E/nodes.$PBS_JOBID > $E/nodes.$a
  local h=$(head -1 $E/nodes.$a)
  ( while true; do mpiexec -np 1 --hosts $h --cpu-bind=none bash -c "export PATH=$XS/bin:\$PATH LD_LIBRARY_PATH=$XS/lib:$XS/lib64:\$LD_LIBRARY_PATH; for g in 0 1 2 3 4 5; do xpu-smi stats -d \$g 2>/dev/null | grep -i 'memory used'; done" >> $d/mem.log 2>&1; echo "--- $(date -u +%FT%TZ)" >> $d/mem.log; sleep 120; done ) & local mp=$!
  r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
  ts="time/tlim=$(head -1 $E/STAGES_$a | awk '{print $1}')"
  echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID nodes $nn from ${r:-t=0} $ts bin $(sha256sum $X | cut -c1-8)" >> $E/runs.log
  local n0=$(cat out.log 2>/dev/null | wc -l)
  mpiexec -np $(( 12*nn )) -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
    $X -d $d -t $WT ${r:+-r $r} -i $E/$a.par $ts >> out.log 2>> err.log
  local rc=$?
  kill $mp 2>/dev/null
  local term=$(tail -n +$(( n0 + 1 )) out.log | grep -o 'Terminating on [a-z -]*' | tail -1)
  echo "$a rc=$rc ${term:-no termination} last=$(tail -n +$(( n0 + 1 )) out.log | grep -o 'cycle=[0-9]* time=[0-9.e+]*' | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
  case "$term" in
    *"time limit"*) stage_done $a ;;
    *"wall"*) ;;
    *) touch $E/STOP_$a; echo "$a STOPPED (no normal termination; rc=$rc)" >> $E/runs.log ;;
  esac
}
nn=$(wc -l < $E/nodes.$PBS_JOBID)
i=1
while read -r a k; do
  [ -z "$a" ] || [ "${a:0:1}" = "#" ] || [ -e $E/STOP_$a ] && continue
  [ $(( i + k - 1 )) -le $nn ] || { echo "$(date -u +%FT%TZ) $a skipped: needs $k nodes, $((nn - i + 1)) left" >> $E/runs.log; continue; }
  ( arm $a $k $i ) < /dev/null & i=$(( i + k ))
done < $E/ARMS
wait

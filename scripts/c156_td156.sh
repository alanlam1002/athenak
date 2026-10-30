#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c156_mombg/td156.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c156_mombg/td156.err
# c156 CPU gates (serial strict ICX builds, one debug node, all cases concurrent):
#   R-177 item 1 (KCLOSE regrid fix): jF = the fixed binary on the moving-bed AMR deck (vs c154 jA0, pre-fix)
#   R-178 (killing_momentum_bg): Toff_new/Toff_mom (off = bitwise), Ton_mom (face_diff + the ghost/owner probe),
#   Vnew/Vmom (Valencia untouched)
set -u
module load oneapi/release/2025.3.1 >/dev/null 2>&1
E=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c156_mombg/cpu
cd $E
F=$E/athena_kdtg_fix_icx; M=$E/athena_mombg_icx
./runj.sh jF $F Mm_kf3a.par time/nlim=3 &
./runj.sh Toff_new $F Mt_t1b.par &
./runj.sh Toff_mom $M Mt_t1b.par &
./runj.sh Ton_mom $M Mt_t1b.par dyn_grmhd/killing_momentum_bg=face_diff dyn_grmhd/killing_probe_cycles=6 &
./runj.sh Vnew $F Mm_v3s.par &
./runj.sh Vmom $M Mm_v3s.par &
wait
echo "td156 done $(date -u +%FT%TZ)" >> $E/runs.log

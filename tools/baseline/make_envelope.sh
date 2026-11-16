#!/bin/bash
# Usage: ./make_envelope.sh N M  -> runs M one-ulp Fortran members at N columns and writes noise_floor_N{N}_m{M}.json
set -e
cd "$(dirname "$0")"
N=$1; M=$2
source /usr/share/lmod/lmod/init/bash; module load intel/2020Update4
export PATH=$INTEL_BIN:$PATH
[ -f ref_${N}_O2.bin ] || ./noah39_driver case_$N.bin ref_${N}_O2.bin 1 >/dev/null
for m in $(seq 1 $M); do
  python3 gen_case.py $N 20261021 $m >/dev/null
  ./noah39_driver case_${N}_m$m.bin ref_${N}_m$m.bin 1 >/dev/null
  rm -f case_${N}_m$m.bin
done
python3 noise_floor_env.py $M noise_floor_N${N}_m${M}.json $N

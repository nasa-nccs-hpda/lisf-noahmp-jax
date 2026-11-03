#!/bin/bash
# Submit every case of the evidence set as SLURM jobs, in parallel across nodes, and one dependent job that compares and writes one summary table.
#   pair_clsm_1d  : EXCLUSIVE V100 node; Fortran LIS (GNU 14, serial, one core) then JAX on the SAME node, strictly sequential. CLSM F2.5, CONUS 0.125 deg, 1 day.
#   pair_hymap_2d : EXCLUSIVE V100 node; Fortran then JAX. Noah-MP 4.0.1 + HyMAP2 routing, CONUS, 2 days.
#   pair_n33_2d   : EXCLUSIVE V100 node; Fortran then JAX. Noah 3.3, CONUS, 2 days.
#   pair_n32_2d   : EXCLUSIVE V100 node; Fortran then JAX. Noah 3.2, CONUS, 2 days.
#   pair_n271_2d  : EXCLUSIVE V100 node; Fortran then JAX. Noah 2.7.1 (derived tables), CONUS, 2 days.
#   pair_mp36_2d  : EXCLUSIVE V100 node; Fortran then JAX. Noah-MP 3.6, CONUS, 2 days.
#   pair_mos_2d   : EXCLUSIVE V100 node; Fortran then JAX. Mosaic LSM, CONUS, 2 days.
#   jax_mp_7d     : V100, not exclusive; JAX only, 7 days Noah-MP CONUS, compared with the stored Fortran reference (conus_mp7/lisout; its wall time is NOT remeasured here).
#   summary       : runs after ALL of the above end (afterany, so a failed case is reported, not dropped); runs lisjax/compare_hist.py per case and writes SUMMARY.md (gpu/summarize_variants.py).
# Timing pairs must not share a node with anything else. Repeats are not bitwise in time: read p99 differences, not equality (Reports/clsm_sensitivity_20261022).
# usage:  gpu/run_variants.sh [NAME]          DRY_RUN=1 writes the batch files and prints them, submits nothing.
# Required environment (see DATA_AND_PATHS.md):
#   LISF_RUN (dev/lisf_run: case directories and the Fortran binaries), LISF_ENV (env_lis.sh of the GNU 14 build), CONDA_BIN (path of the conda executable)
# Optional: JAX_ENV (conda env name, default graphcast-env), LISF_PROJ (default: this checkout), SBATCH_HOST (e.g. gpulogin1: submit over ssh when the local host has no SLURM client),
#   PARTITION (compute), GPU_CONSTRAINT (v100), ONLY="pair_clsm_1d ..." (submit a subset; the summary still runs), TIME_PAIR/TIME_JAX (HH:MM:SS).
set -u
P=${LISF_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
for v in LISF_RUN LISF_ENV CONDA_BIN; do
  [ -n "${!v:-}" ] || { echo "run_variants.sh: set $v (see DATA_AND_PATHS.md)" >&2; exit 2; }
done
for f in "$LISF_RUN/LIS_clsm_dump2" "$LISF_RUN/LIS_mp_hy" "$LISF_RUN/conus_clsm1d/lis.config" "$LISF_RUN/conus_hy/lis.config" "$LISF_RUN/conus_n33_2d/lis.config" "$LISF_RUN/conus_mos_2d/lis.config" "$LISF_RUN/conus_n32_2d/lis.config" "$LISF_RUN/conus_n271_2d/lis.config" "$LISF_RUN/conus_mp36_2d/lis.config" "$LISF_RUN/conus_mp7/lisout" "$LISF_ENV" "$CONDA_BIN"; do
  [ -e "$f" ] || { echo "run_variants.sh: missing $f" >&2; exit 2; }
done
ENVN=${JAX_ENV:-graphcast-env}; PART=${PARTITION:-compute}; CONS=${GPU_CONSTRAINT:-v100}
NAME=${1:-variants_$(date +%Y%m%d_%H%M)}
RUN=$P/gpu_runs/$NAME; JOBS=$RUN/jobs; mkdir -p "$JOBS"
ONLY=${ONLY:-pair_clsm_1d pair_hymap_2d pair_n33_2d pair_n32_2d pair_n271_2d pair_mp36_2d pair_mos_2d jax_mp_7d}

submit() {   # submit <batchfile> [dependency-spec]  -> prints the job id
  local f=$1 dep=${2:-}
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_$(basename "$f" .sbatch)"; return; fi
  local cmd="sbatch --parsable ${dep:+--dependency=$dep }$f" out
  if [ -n "${SBATCH_HOST:-}" ]; then out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=5 "$SBATCH_HOST" "$cmd" 2>/dev/null); else out=$($cmd 2>/dev/null); fi
  out=$(printf '%s\n' "$out" | sed 's/;.*//' | grep -E '^[0-9]+$' | tail -1)
  [ -n "$out" ] || { echo "run_variants.sh: sbatch failed for $f (no job id returned)" >&2; return 1; }
  echo "$out"
}
header() {   # header <jobname> <time> <extra #SBATCH lines...>
  local nm=$1 tm=$2; shift 2
  cat <<HDR
#!/bin/bash
#SBATCH --job-name=$nm
#SBATCH --partition=$PART
#SBATCH --time=$tm
#SBATCH --output=$RUN/$nm.out
$(for l in "$@"; do echo "#SBATCH $l"; done)
export LISF_RUN="$LISF_RUN" LISF_ENV="$LISF_ENV" LISF_PROJ="$P" OMP_NUM_THREADS=1
echo "host: \$(hostname) gpu: \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1) cpu: \$(lscpu | grep 'Model name' | sed 's/.*: *//')"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
HDR
}
GPU_EXCL=(--constraint=$CONS --gres=gpu:1 --nodes=1 --cpus-per-task=8 --mem=96G --exclusive)
GPU_SHARED=(--constraint=$CONS --gres=gpu:1 --nodes=1 --cpus-per-task=8 --mem=96G)
JAXRUN="$CONDA_BIN run --no-capture-output -n $ENVN python -u"

pair_batch() {   # pair_batch <name> <case dir> <fortran bin> <fortran cfg> <fortran outdir> <fortran env> <jax command>
  local nm=$1 case=$2 bin=$3 cfg=$4 fout=$5 fenv=$6 jaxcmd=$7 f=$JOBS/$1.sbatch
  { header "$nm" "${TIME_PAIR:-03:00:00}" "${GPU_EXCL[@]}"
    cat <<BODY
echo "=== Fortran LIS, GNU 14 serial, one core"
( source "\$LISF_ENV"; cd "\$LISF_RUN/$case"; rm -rf $fout; mkdir -p $fout; $fenv
  sed 's#\./lisout#./$fout#g' $cfg > lis.config.$nm
  /usr/bin/time -v "\$LISF_RUN/$bin" -f lis.config.$nm > lis.$nm.out 2> lis.$nm.time
  grep -E "Elapsed|Maximum resident|User time" lis.$nm.time; tail -1 $fout/lislog.0000 )
echo "=== JAX on the same node"
cd "\$LISF_PROJ"
$jaxcmd
BODY
  } > "$f"; echo "$f"
}

ids=""
for v in $ONLY; do
  case $v in
    pair_clsm_1d)
      f=$(pair_batch pair_clsm_1d conus_clsm1d LIS_clsm_dump2 lis.config lisout_$NAME "export JAXDUMP_CLSM_MAXSTEP=0" \
        "$JAXRUN -m lisjax.run_clsm \"\$LISF_RUN/conus_clsm1d\" \"\$LISF_RUN/conus_clsm1d/jax_$NAME\" lis.config 2017020100 2017020200") ;;
    pair_hymap_2d)
      f=$(pair_batch pair_hymap_2d conus_hy LIS_mp_hy lis.config lisout_$NAME "" \
        "$JAXRUN lisjax/bench_conus.py \"\$LISF_RUN/conus_hy\" \"\$LISF_RUN/conus_hy/jax_$NAME\" 2") ;;
    pair_n33_2d)
      f=$(pair_batch pair_n33_2d conus_n33_2d LIS_clsm_dump2 lis.config lisout_$NAME "export JAXDUMP_CLSM_MAXSTEP=0" \
        "$JAXRUN lisjax/run.py \"\$LISF_RUN/conus_n33_2d\" \"\$LISF_RUN/conus_n33_2d/jax_$NAME\" lis.config 20170201 20170203") ;;
    pair_n32_2d)
      f=$(pair_batch pair_n32_2d conus_n32_2d LIS_clsm_dump2 lis.config lisout_$NAME "export JAXDUMP_CLSM_MAXSTEP=0" \
        "$JAXRUN lisjax/run.py \"\$LISF_RUN/conus_n32_2d\" \"\$LISF_RUN/conus_n32_2d/jax_$NAME\" lis.config 20170201 20170203") ;;
    pair_n271_2d)
      f=$(pair_batch pair_n271_2d conus_n271_2d LIS_clsm_dump2 lis.config lisout_$NAME "export JAXDUMP_CLSM_MAXSTEP=0" \
        "$JAXRUN lisjax/run.py \"\$LISF_RUN/conus_n271_2d\" \"\$LISF_RUN/conus_n271_2d/jax_$NAME\" lis.config 20170201 20170203") ;;
    pair_mp36_2d)
      f=$(pair_batch pair_mp36_2d conus_mp36_2d LIS_mp_clean lis.config lisout_$NAME "" \
        "(cd \"\$LISF_RUN/conus_mp36_2d\"; sed 's#\\./lisout#./jax_$NAME#g' lis.config > lis.config.jax_$NAME); $JAXRUN -m lisjax.lis -C \"\$LISF_RUN/conus_mp36_2d\" \"\$LISF_RUN/conus_mp36_2d/lis.config.jax_$NAME\"") ;;
    pair_mos_2d)
      f=$(pair_batch pair_mos_2d conus_mos_2d LIS_clsm_dump2 lis.config lisout_$NAME "export JAXDUMP_CLSM_MAXSTEP=0" \
        "$JAXRUN -m lisjax.run_mos \"\$LISF_RUN/conus_mos_2d\" \"\$LISF_RUN/conus_mos_2d/jax_$NAME\" lis.config 2017020100 2017020300") ;;
    jax_mp_7d)
      f=$JOBS/jax_mp_7d.sbatch
      { header jax_mp_7d "${TIME_JAX:-01:30:00}" "${GPU_SHARED[@]}"
        echo "cd \"\$LISF_PROJ\""
        echo "$JAXRUN lisjax/bench_conus.py \"\$LISF_RUN/conus_mp7\" \"\$LISF_RUN/conus_mp7/jax_$NAME\" 7"; } > "$f" ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
  id=$(submit "$f") || exit 1
  echo "$v -> job $id"; ids="$ids:$id"
done

S=$JOBS/summary.sbatch
{ header summary "00:30:00" --cpus-per-task=4 --mem=16G
  echo "cd \"\$LISF_PROJ\""
  echo "JAX_PLATFORMS=cpu $CONDA_BIN run --no-capture-output -n $ENVN python gpu/summarize_variants.py \"$RUN\" \"\$LISF_RUN\" \"$NAME\" > \"$RUN/SUMMARY.md\" 2>&1; cat \"$RUN/SUMMARY.md\""; } > "$S"
sid=$(submit "$S" "${ids:+afterany$ids}") || exit 1
echo "summary -> job $sid (writes $RUN/SUMMARY.md)"

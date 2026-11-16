#!/bin/bash
#SBATCH --job-name=intelx
#SBATCH --partition=compute
#SBATCH --account=ilab
#SBATCH --constraint=mil
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# R's Intel-built LIS 7.8, R's Noah-MP 5.0 NLDAS-2 case, NetCDF NLDAS-2.0 reader on GRIB-converted files, 1-6 Oct 2008, at 20, 32 and 40 ranks.
J=$DISCOVER_WORK/jaxtest
R=$R_CASE
G=/discover/nobackup/projects/lis/MET_FORCING/NLDAS2.FORCING
X=$J/intel
mkdir -p $X; cd $X
ln -sfn $R/input input; ln -sfn $R/lis_input.nldas2.noahmp401.OL.nc lis_input.nldas2.noahmp401.OL.nc; ln -sfn $R/LDT LDT
module load python/GEOSpyD/24.11.3-0/3.12
if [ ! -d $X/nc/2008/280 ]; then $J/venv2/bin/python $J/lisf-noahmp-jax/tools/forcing/grib_to_nc.py $G $X/nc 20081001 20081006 16; fi
module purge
# the module file is the one of the public LISF repo (env/discover/sles15); copy kept here because the module path of R's account is not visible from ours
module use $J/lisf-noahmp-jax/scripts/discover/intel_xcheck/modulefiles
module load lisf_7.8_intel_2025.3.2
ulimit -s unlimited
for cfg in "20 4 5" "32 8 4" "40 8 5"; do
  set -- $cfg; N=$1
  rm -rf out_$N; mkdir -p out_$N
  python3 $J/lisf-noahmp-jax/scripts/discover/intel_xcheck/mkcfg_intel.py $R/lis.config.nldas2.Basic.OL cfg_$N $X/nc $X/out_$N $2 $3
  echo "== $N ranks start $(date +%T)"
  /usr/bin/time -v mpirun -n $N $R/LIS --file cfg_$N > lis_$N.out 2> time_$N.txt
  echo "exit $? ; $(grep Elapsed time_$N.txt)"
  ls out_$N/SURFACEMODEL/200810 2>/dev/null | wc -l
done
echo XCHECKDONE

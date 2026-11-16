#!/bin/bash
# Discover LOGIN node, step 1 of the tarball check (compute nodes cannot reach PyPI, so the environment is built here).
# Expects $DISCOVER_WORK/tbcheck/lisf-noahmp-jax-*.tar.gz. Builds the locked environment (CPU, GPU and GRIB extras) in tbcheck/uvenv.
# usage: bash tarball_check_setup.sh      then: sbatch tarball_check_cpu.sbatch ; sbatch -C rome tarball_check_a100.sbatch
set -e
W=$DISCOVER_WORK/tbcheck; cd $W
module load python/GEOSpyD/24.11.3-0/3.12
rm -rf lisf-noahmp-jax uvboot uvenv; tar xzf lisf-noahmp-jax-*.tar.gz
python3 -m venv uvboot && uvboot/bin/pip install -q uv==0.13.7
cd lisf-noahmp-jax
UV_PROJECT_ENVIRONMENT=$W/uvenv $W/uvboot/bin/uv sync --locked --extra test --extra gpu --extra grib
$W/uvenv/bin/python -c "import jax, netCDF4, pygrib; print('jax', jax.__version__, 'pygrib', pygrib.__version__)"
echo SETUP_DONE

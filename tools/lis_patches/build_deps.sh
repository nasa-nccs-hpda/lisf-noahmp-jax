#!/bin/bash
# Build HDF5, netCDF-C, netCDF-Fortran, ecCodes and ESMF (mpiuni) with GNU 14 into $DEPS/install.
set -e
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_gnu14.sh
cd $DEPS/src; P=$DEPS/install; mkdir -p $P; J=8
step() { echo "=== $(date +%T) $1"; }
if [ ! -f $P/lib/libhdf5.so ]; then step hdf5; tar xzf hdf5-1.14.4-3.tar.gz; cd hdf5-1.14.4-3; ./configure --prefix=$P --enable-fortran --enable-shared >../hdf5.log 2>&1; make -j$J >>../hdf5.log 2>&1; make install >>../hdf5.log 2>&1; cd ..; fi
if [ ! -f $P/lib/libnetcdf.so ]; then step netcdf-c; tar xzf netcdf-c-4.9.2.tar.gz; cd netcdf-c-4.9.2; CPPFLAGS=-I$P/include LDFLAGS=-L$P/lib ./configure --prefix=$P --enable-netcdf-4 --disable-dap --disable-byterange >../netcdfc.log 2>&1; make -j$J >>../netcdfc.log 2>&1; make install >>../netcdfc.log 2>&1; cd ..; fi
if [ ! -f $P/lib/libnetcdff.so ]; then step netcdf-fortran; tar xzf netcdf-fortran-4.6.1.tar.gz; cd netcdf-fortran-4.6.1; CPPFLAGS=-I$P/include LDFLAGS=-L$P/lib ./configure --prefix=$P >../netcdff.log 2>&1; GCCROOT=$(dirname $(dirname $(which gfortran))); find . -name Makefile -o -name libtool -o -name config.status | xargs sed -i "s#/app/gcc/platform/x86_64/rhel/9.4/14.2.0#$GCCROOT#g"; make -j$J >>../netcdff.log 2>&1; make install >>../netcdff.log 2>&1; cd ..; fi
if [ ! -f $P/lib64/libeccodes.so ] && [ ! -f $P/lib/libeccodes.so ]; then step eccodes; tar xzf eccodes-2.38.3.tar.gz; mkdir -p eccodes-build; cd eccodes-build; cmake ../eccodes-2.38.3 -DCMAKE_INSTALL_PREFIX=$P -DENABLE_FORTRAN=ON -DENABLE_NETCDF=OFF -DENABLE_JPG=OFF -DENABLE_AEC=OFF -DENABLE_PYTHON=OFF -DENABLE_MEMFS=OFF -DCMAKE_Fortran_COMPILER=gfortran >../eccodes.log 2>&1; make -j$J >>../eccodes.log 2>&1; make install >>../eccodes.log 2>&1; cd ..; fi
step esmf
cd esmf-8.8.0
export ESMF_DIR=$PWD ESMF_COMPILER=gfortran ESMF_COMM=mpiuni ESMF_BOPT=O ESMF_INSTALL_PREFIX=$P/esmf ESMF_NETCDF=nc-config ESMF_NETCDF_INCLUDE=$P/include ESMF_NETCDF_LIBPATH=$P/lib ESMF_NETCDF_LIBS="-lnetcdff -lnetcdf" ESMF_PIO=OFF
export ESMF_OS=Linux
make -j$J >../esmf.log 2>&1
make install >>../esmf.log 2>&1
step done

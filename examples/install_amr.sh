git clone --recursive https://github.com/exawind/amr-wind.git
mkdir amr-wind/build && cd amr-wind/build

module purge
ml intel-oneapi-mpi
ml netcdf/4.9.2-intel-oneapi-mpi-intel
cmake -DAMR_WIND_ENABLE_MPI:BOOL=ON -DAMR_WIND_ENABLE_NETCDF:BOOL=ON -DAMR_WIND_ENABLE_TESTS:BOOL=ON ../
make -j10
ctest --output-on-failure

#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=104
#SBATCH --time=01:00:00
#SBATCH --output=out.%x_%j

module purge
#module load craype-x86-spr
#module load perftools-base/23.12.0
#ml cray-dsmml/0.2.2
#ml PrgEnv-intel/8.5.0
#ml libfabric/1.15.2.0
#ml intel-oneapi-tbb/2021.11.0-intel
#ml intel-oneapi-mkl/2024.0.0-intel
ml intel-oneapi-mpi
ml netcdf/4.9.2-intel-oneapi-mpi-intel

#export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel:$LD_LIBRARY_PATH
#export MPICH_OFI_CQ_STALL=1
#export MPICH_OFI_CQ_STALL_USECS=16

#PPN=104
#RANKS=$((${SLURM_JOB_NUM_NODES}*${PPN}))

input_file=test_precursor.inp

#PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/build

#amr_exec=$(which amr_wind)
amr_exec=/home/ahenry/toolboxes/whoc_enc/amr-wind/build/amr_wind

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. tasks     = $SLURM_NTASKS"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"
echo "amr-wind executable = $amr_exec"

srun --distribution=cyclic:cyclic --cpu_bind=cores amr_wind $input_file

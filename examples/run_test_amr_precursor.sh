#!/bin/bash
#SBATCH --account=ssc
#SBATCH --nodes=8
#SBATCH --time=48:00:00
#SBATCH --output=out.%x_%j

module purge
ml PrgEnv-intel
ml cray-libsci
ml netcdf/4.9.2-intel-oneapi-mpi-intel

#export SPACK_MANAGER="/home/ahenry/toolboxes/spack-manager"
#source $SPACK_MANAGER/start.sh
#spack-start
#quick-activate /home/ahenry/toolboxes/whoc_env
#spack load amr-wind

export OMP_PROC_BIND=spread
export KMP_AFFINITY=balanced
ranks_per_node=92
mpi_ranks=736 #1472
input_file=test_precursor.inp
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy
amr_exec=$(which amr_wind)
echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. MPI Ranks = $mpi_ranks"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"
echo "amr-wind executable = $amr_exec"
srun -n $mpi_ranks -c 1 --cpu_bind=cores amr_wind $input_file

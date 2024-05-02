#!/bin/bash
#SBATCH --job-name=amr_precursor
#SBATCH --time=12:00:00
##SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

# A lot of modules and conda stuff
module purge

export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier
export MPICH_COLL_OPT_OFF=mpi_allreduce

export SPACK_MANAGER="/home/ahenry/toolboxes/spack-manager"
source $SPACK_MANAGER/start.sh
spack-start
quick-activate /home/ahenry/toolboxes/whoc_env
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy
spack load amr-wind+helics+openfast

echo "Starting AMR-Wind job at: " $(date)
echo $SLURM_NTASKS
# Now go back to scratch folder and launch the job
srun -n $SLURM_NTASKS /home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy/amr_wind amr_precursor_original.inp
echo "Finished AMR-Wind job at: " $(date)

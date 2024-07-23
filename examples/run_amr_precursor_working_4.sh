#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=104
#SBATCH --time=60:00:00
#SBATCH --output=out.%x_%j

module purge
ml intel-oneapi-mpi
ml netcdf/4.9.2-intel-oneapi-mpi-intel

input_file=amr_precursor_working_4.inp

PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/build

amr_exec=/home/ahenry/toolboxes/whoc_enc/amr-wind/build/amr_wind

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. tasks     = $SLURM_NTASKS"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"
echo "amr-wind executable = $amr_exec"

srun --distribution=cyclic:cyclic --cpu_bind=cores amr_wind $input_file

if [ -d "/projects/ssc/ahenry/whoc/amr_precursors/post_processing_4" ]; then
	rm -rf /projects/ssc/ahenry/whoc/amr_precursors/post_processing_4
fi
mkdir /projects/ssc/ahenry/whoc/amr_precursors/post_processing_4
mv ./post_processing/* /projects/ssc/ahenry/whoc/amr_precursors/post_processing_4


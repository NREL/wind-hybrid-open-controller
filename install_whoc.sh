ssh ahenry@kestrel.hpc.nrel.gov
ml mamba
mamba create --prefix=/projects/ssc/ahenry/conda/envs/whoc --y
# conda create --prefix=/projects/aohe7145/software/anaconda/envs/whoc --y
mamba activate whoc

git clone https://github.com/achenry/wind-hybrid-open-controller.git
cd wind-hybrid-open-controller && git checkout feature/mpc && python setup.py develop && cd ..
git clone https://github.com/NREL/floris.git
git clone https://github.com/achenry/hercules.git
cd hercules && git checkout develop && cd ..
git clone https://github.com/NREL/moa_python.git

python -m pip install -e wind-hybrid-open-controller
python -m pip install -e floris
python -m pip install -e hercules
python -m pip install -e moa_python

# mamba install memory_profiler # matplotlib openmpi
mamba install -c conda-forge mpi4py pyoptsparse pandas pyyaml memory_profiler seaborn --y 
#conda install -c nrel nrel-pysam

python -m pip install https://github.com/NREL/SEAS/blob/main/SEAS.tar.gz?raw=true
python -m pip install git+https://github.com/NREL/electrolyzer.git

# conda update libstdcxx-ng
# conda install -c conda-forge libstdcxx-ng 
# conda install -c conda-forge libstdcxx-ng --force-reinstall
export LD_LIBRARY_PATH=/projects/aohe7145/software/anaconda/envs/whoc/lib
python run_case_studies.py 0 -rs -st 480 -ns 1 -m mpi -sd /projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies
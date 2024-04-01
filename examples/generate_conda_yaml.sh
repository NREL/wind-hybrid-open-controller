conda env export --no-builds > ./examples/whoc.yml -p /home/ahenry/.conda-envs/whoc
# NOTE: must remove libgfortran line from whoc.yml then run `conda env create --name whoc --file whoc.yml``
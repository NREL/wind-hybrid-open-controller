# Wind Hybrid Open Controller

The Wind Hybrid Open Controller (WHOC) is a python-based tool for real-time 
plant-level wind farm control and wind-based hybrid plant control.
WHOC will primarily be run in simulation, although we intend that it could be 
used for physical plants in future. 

WHOC will provide simple farm-level (and hybrid plant-level) controls such as
wake steering control, spatial filtering/consensus, active power control, 
and coordinated control of hyrbid power plant assets;
and create an entry point for the development of more advanced controllers. 

WHOC will interface with various simulation testbeds and lower level 
controllers, including:
- [Hercules](https://github.com/NREL/hercules)
- [FAST.Farm](https://github.com/OpenFAST/openfast)
- [ROSCO](https://github.com/NREL/rosco)

WHOC controllers will also call on design tools such as
[FLORIS](https://github.com/NREL/floris).

## Code development
To contribute to WHOC, please consider forking the main github repository,
with the [main repo](github.com/NREL/wind-hybrid-open-controller) as an 
upstream remote. To submit a new feature or bug fix, create a new branch 
in your fork and submit a pull request back to the `develop` branch in the 
main repo. The pull request will be reviewed by other WHOC developers and 
merged (using "squash and merge") into the `develop` branch. Periodically, 
the `develop` branch will be merged into the `main` branch and a version 
number will be assigned.
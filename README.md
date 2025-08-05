# Wind Hybrid Open Controller

The Wind Hybrid Open Controller (WHOC) is a python-based tool for real-time 
plant-level wind farm control and wind-based hybrid plant control.
WHOC will primarily be run in simulation, although we intend that it could be 
used for physical plants in future. 

WHOC will provide simple farm-level (and hybrid plant-level) controls such as
wake steering control, spatial filtering/consensus, active power control, 
and coordinated control of hybrid power plant assets;
and create an entry point for the development of more advanced controllers. 

WHOC will interface with various simulation testbeds and lower level 
controllers, including:
- [Hercules](https://github.com/NREL/hercules)
- [FAST.Farm](https://github.com/OpenFAST/openfast)
- [ROSCO](https://github.com/NREL/rosco)

WHOC controllers will also call on design tools such as
[FLORIS](https://github.com/NREL/floris).

## WETO software

WHOC is primarily developed with the support from the U.S. Department of Energy and
is part of the [WETO Software Stack](https://nrel.github.io/WETOStack).
For more information and other integrated modeling software, see:

- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Wind Farm Controls Workshop](https://www.youtube.com/watch?v=f-w6whxIBrA&list=PL6ksUtsZI1dwRXeWFCmJT6cEN1xijsHJz)

NREL's software record for WHOC is SWR-25-54.

## Documentation

Documentation for WHOC, including installation instructions, can be found
[here](https://nrel.github.io/wind-hybrid-open-controller/intro.html).

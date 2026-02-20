# Hycon

Hycon is a python-based tool for real-time hybrid power plant control.
Hycon will primarily be run in simulation, although we intend that it could be 
used for physical plants in future. 

Hycon provides plant-level (and hybrid plant-level) controls such as
wake steering control, spatial filtering/consensus, active power control, 
and coordinated control of hybrid power plant assets;
and creates an entry point for the development of more advanced controllers. 

Hycon will interface with various simulation testbeds and lower level 
controllers, including:
- [Hercules](https://github.com/NatLabRockies/hercules)
- [FAST.Farm](https://github.com/OpenFAST/openfast)
- [ROSCO](https://github.com/NatLabRockies/rosco)

Hycon controllers will also call on design tools such as
[FLORIS](https://github.com/NatLabRockies/floris).

## WETO software

Hycon is primarily developed with the support from the U.S. Department of Energy and
is part of the [WETO Software Stack](https://natlabrockies.github.io/WETOStack).
For more information and other integrated modeling software, see:

- [Portfolio Overview](https://natlabrockies.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://natlabrockies.github.io/WETOStack/_static/entry_guide/index.html)
- [Wind Farm Controls Workshop](https://www.youtube.com/watch?v=f-w6whxIBrA&list=PL6ksUtsZI1dwRXeWFCmJT6cEN1xijsHJz)

NLR's software record for Hycon is SWR-25-54.

## Documentation

Documentation for Hycon, including installation instructions, can be found
[here](https://natlabrockies.github.io/hycon/intro.html).

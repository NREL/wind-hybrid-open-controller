# Controllers

The `whoc.controllers` module contains a library of wind and hybrid power plant
controllers. Each controller must inherit from `ControllerBase` (see 
controller_base.py) and implement a
mandatory `compute_controls()` method, which contains the relevant control 
algorithm and writes final control signals to the `controls_dict` attribute 
as key-value pairs. `compute_controls()` is, in turn, called in the `step()`
method of `ControllerBase`.

## Available controllers

### LookupBasedWakeSteeringController
Yaw controller that implements wake steering based on a lookup table. 
<<<<<<< HEAD
Requires a df_opt object produced by a FLORIS yaw optimization routine. See example 
=======
Requires a `df_opt` object produced by a FLORIS yaw optimization routine. See example 
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824
lookup-based_wake_steering_florisstandin for example usage.

Currently, yaw angles are set based purely on the (local turbine) wind direction. The lookup table
is sampled at a hardcoded wind speed of 8 m/s. This will be updated in future when an interface is
developed for a simulator that provides wind turbine wind speeds also.

### WakeSteeringROSCOStandin
<<<<<<< HEAD
May be combined into a universal simple wake steeringcontroller.

### HerculesWindBatteryController
TO WRITE
=======
Not yet developed. May be combined into a universal simple LookupBasedWakeSteeringController.

### WindBatteryController
Placeholder for a controller that manages both a wind power plant and colocated
battery.

### WindFarmPowerDistributingController

Wind farm-level power controller that simply distributes a farm-level power 
reference between wind turbines evenly, without checking whether turbines are 
able to produce power at the requested level. Not expected to perform well when
wind turbines are waked or cannot produce the desired power for other reasons. 
However, is a useful comparison case for the WindFarmPowerTrackingController 
(described below).

### WindFarmPowerTrackingController

Closed-loop wind farm-level power controller that distributes a farm-level 
power reference among the wind turbines in a farm and adjusts the requests made
from each turbine depending on whether the power reference has been met. 
Developed under the [A2e2g project](https://github.com/NREL/a2e2g), with 
further details provided in 
[Sinner et al.](https://pubs.aip.org/aip/jrse/article/15/5/053304/2913100).

Integral action, as well as gain scheduling based on turbine saturation, has been disabled as 
simple proportional control appears sufficient currently. However, these may be enabled at a 
later date if needed. The `proportional_gain` for the controller may be provided on instantiation,
and defaults to `proportional_gain = 1`.
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824

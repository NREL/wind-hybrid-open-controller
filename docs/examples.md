# Examples

<<<<<<< HEAD
The `examples` subdirectoy contains a series of examples that can be run to test the functionality
=======
The `examples` subdirectory contains a series of examples that can be run to test the functionality
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824
of certain controllers and interfaces.

### lookup-based_wake_steering_florisstandin
2-turbine example of lookup-based wake steering control, run using Hercules with the FLORIS standin
in place of AMR-Wind for exposition purposes. To run this example, navigate to the 
<<<<<<< HEAD
examples/lookup-based_wake_steering_florisstandin and then run the following.
=======
examples/lookup-based_wake_steering_florisstandin folder and then run the following.
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824
```
python construct_yaw_offsets.py
```

<<<<<<< HEAD
Not that, currently, contruct_yaw_offsets.py requires FLORIS version 3, whereas the rest of the 
example requires FLORIS version 4. As a result, we provide the generated offsets in
yaw_offesets.pkl. To avoid regenerating yaw_offsets.pkl (and therefore avoid the current need for 
=======
Not that, currently, construct_yaw_offsets.py requires FLORIS version 3, whereas the rest of the 
example requires FLORIS version 4. As a result, we provide the generated offsets in
yaw_offsets.pkl. To avoid regenerating yaw_offsets.pkl (and therefore avoid the current need for 
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824
floris v3), set `optimize_yaw_offsets = False` at the beginning of construct_yaw_offsets.py before
running.

Next, run
```
./bash_script.sh
```
You will need to have and up-to-date Hercules (possibly on the develop branch) in your conda
environment to run this. You may also need to changed permissions to run bash_script.sh as an 
executable (`chmod +x bash_script.sh`).

Finally, run the post-processing script
```
python plot_output_data.py
```
This should produce the following plot.
![Results of lookup-based_wake_steering_florisstandin example](
    graphics/lookup-table-example-plot.png
<<<<<<< HEAD
=======
)

## wind_farm_power_tracking_florisstandin
2-turbine example of wind-farm-level power reference tracking, run using Hercules with the FLORIS 
standin in place of AMR-Wind for exposition purposes. To run this example, navigate to the 
examples/wind_farm_power_tracking_florisstandin folder and run the following:
```
./bash_script.sh
```

This will run both a closed-loop controller, which compensates for underproduction at individual 
turbines, and an open-loop controller, which simply distributes the farm-wide reference evenly
amongst the turbines of the farm without feedback. The resulting trajectories are plotted, 
producing:
![Results of wind_farm_power_tracking_florisstandin example](
    graphics/wf-power-tracking-plot.png
>>>>>>> 3caa5f54c338e875c21730507adab5c4c0aec824
)
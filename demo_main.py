import numpy as np
import matplotlib.pyplot as plt

from whoc.interfaces.hercules_actuator_disk_yaw_interface import WHOC_AD_yaw_connection
from whoc.controllers.wake_steering_actuatordisk_standin import WakeSteeringADStandin

demo_hercules_dict = {
    "dt":1.0,
    "hercules_comms":{
        "amr_wind":{
            "wind_farm_0":{
                "type": "amr_wind_local",
                "amr_wind_input_file": "amr_input.inp",
            }
        }
    },
    "controller":{
        "num_turbines":2,
        "initial_conditions":{
            "yaw":[10.0, 15.0]
        }
    }
}

interface = WHOC_AD_yaw_connection(demo_hercules_dict)

controller = WakeSteeringADStandin(
    interface,
    demo_hercules_dict
)

# Create a little loop to demonstrate how the controller works
wd_base = np.linspace(280, 300, 50)
np.random.seed(0)
wind_dir = np.tile(wd_base, (2,1)).T + \
    np.random.normal(scale=5.0, size=(len(wd_base),2))

yaw_angles = []
for i in range(wind_dir.shape[0]):
    demo_hercules_dict["hercules_comms"]\
                      ["amr_wind"]\
                      ["wind_farm_0"]\
                      ["turbine_wind_directions"] = wind_dir[i,:]
    demo_hercules_dict["hercules_comms"]\
                      ["amr_wind"]\
                      ["wind_farm_0"]\
                      ["turbine_wind_speeds"] = 8
    demo_hercules_dict["hercules_comms"]\
                      ["amr_wind"]\
                      ["wind_farm_0"]\
                      ["turbine_powers"] = 2000
    demo_hercules_dict["time"] = float(i)
    
    
    demo_hercules_dict = controller.step(hercules_dict=demo_hercules_dict)

    yaw_angles.append(
        demo_hercules_dict["hercules_comms"]\
                          ["amr_wind"]\
                          ["wind_farm_0"]\
                          ["turbine_yaw_angles"]
    )

yaw_angles = np.array(yaw_angles)
print(yaw_angles)
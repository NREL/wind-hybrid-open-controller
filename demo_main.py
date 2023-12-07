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
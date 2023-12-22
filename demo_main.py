# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import matplotlib.pyplot as plt
import numpy as np
from whoc.controllers.wake_steering_actuatordisk_standin import WakeSteeringADStandin
from whoc.interfaces.hercules_actuator_disk_yaw_interface import HerculesADYawInterface

demo_hercules_dict = {
    "dt": 1.0,
    "hercules_comms": {
        "amr_wind": {
            "wind_farm_0": {
                "type": "amr_wind_local",
                "amr_wind_input_file": "amr_input.inp",
            }
        }
    },
    "controller": {"num_turbines": 2, "initial_conditions": {"yaw": [10.0, 15.0]}},
}

interface = HerculesADYawInterface(demo_hercules_dict)

controller = WakeSteeringADStandin(interface, demo_hercules_dict)

# Create a little loop to demonstrate how the controller works
wd_base = np.linspace(280, 300, 50)
np.random.seed(0)
wind_dir = np.tile(wd_base, (2, 1)).T + np.random.normal(scale=5.0, size=(len(wd_base), 2))

yaw_angles = []
for i in range(wind_dir.shape[0]):
    demo_hercules_dict["hercules_comms"]["amr_wind"]["wind_farm_0"][
        "turbine_wind_directions"
    ] = wind_dir[i, :]
    demo_hercules_dict["hercules_comms"]["amr_wind"]["wind_farm_0"]["turbine_wind_speeds"] = 8
    demo_hercules_dict["hercules_comms"]["amr_wind"]["wind_farm_0"]["turbine_powers"] = 2000
    demo_hercules_dict["time"] = float(i)

    demo_hercules_dict = controller.step(hercules_dict=demo_hercules_dict)

    yaw_angles.append(
        demo_hercules_dict["hercules_comms"]["amr_wind"]["wind_farm_0"]["turbine_yaw_angles"]
    )

yaw_angles = np.array(yaw_angles)
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
for i in range(2):
    ax[i].plot(range(wind_dir.shape[0]), wind_dir[:, i], color="C0", label="wd")
    ax[i].plot(range(wind_dir.shape[0]), yaw_angles[:, i], color="black", label="yaw stpt")
    ax[i].set_ylabel("Direction, T{0} [deg]".format(i))
    ax[i].grid()
ax[1].set_xlabel("Time")
ax[1].set_xlim([0, wind_dir.shape[0]])
ax[0].legend()

plt.show()

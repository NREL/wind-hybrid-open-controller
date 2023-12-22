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

import multiprocessing as mp

from ROSCO_toolbox.control_interface import turbine_zmq_server
from whoc.interfaces._temp_server import sim_rosco

"""
NOTE: this is not yet working.
"""


def run_zmq():
    connect_zmq = True
    s = turbine_zmq_server(network_address="tcp://*:5555", timeout=10.0, verbose=True)
    while connect_zmq:
        #  Get latest measurements from ROSCO
        measurements = s.get_measurements()

        # Decide new control input based on measurements
        current_time = measurements["Time"]
        if current_time <= 10.0:
            yaw_setpoint = 0.0
        else:
            yaw_setpoint = 20.0

            # Send new setpoints back to ROSCO
        s.send_controls(nacelleHeading=yaw_setpoint)

        if measurements["iStatus"] == -1:
            connect_zmq = False
            s._disconnect()


if __name__ == "__main__":
    p1 = mp.Process(target=run_zmq)
    p1.start()
    p2 = mp.Process(target=sim_rosco)
    p2.start()
    p1.join()
    p2.join()

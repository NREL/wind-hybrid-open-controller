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

from abc import ABCMeta, abstractmethod


class ControllerBase(metaclass=ABCMeta):
    def __init__(self, interface, verbose=True):
        self._s = interface
        self.verbose = verbose

        # if use_helics_interface:
        #     raise NotImplementedError(
        #         "HELICS interface has not yet been implemented."
        #     )

        #     # TODO: eventually, this would set up a federate (with same
        #     # public methods as the whoc_zmq_server
        #     #self._s = whoc_helics_federate()

        # elif use_zmq_interface:
        #     from servers.zmq_server import WHOC_zmq_server

        #     # TODO: set up HELICS server
        #     # Set up connections with each turbine
        #     self._s = WHOC_zmq_server(network_address="tcp://*:5555",
        #         timeout=timeout, verbose=True)

        # elif use_direct_hercules_connection:
        #     from servers.direct_hercules_connection import WHOC_AD_yaw_connection
        #     self._s = WHOC_AD_yaw_connection(hercules_dict)

        # else:
        #     from servers.python_server import WHOC_python_server
        #     self._s = WHOC_python_server()

        # Initialize controls to send
        self.controls_dict = None

    def _receive_measurements(self, hercules_dict=None):
        # May need to eventually loop here, depending on server set up.
        self.measurements_dict = self._s.get_measurements(hercules_dict)

        return None

    def _send_controls(self) -> dict:
        # TODO what are other_args for?
        self._s.check_controls(self.controls_dict)
        controller_output = self._s.send_controls(**self.controls_dict)

        return controller_output  # or main_dict, or what?

    def step(self, hercules_dict=None):
        # If not running with direct hercules integration,
        # hercules_dict may simply be None throughout this method.
        self._receive_measurements()

        self.compute_controls() # set self.controls_dict

        observations = self._send_controls()

        return observations  # May simply be None.

    @abstractmethod
    def compute_controls(self):
        # Control algorithms should be implemented in the compute_controls
        # method of the child class.
        pass

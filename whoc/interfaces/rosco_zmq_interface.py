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

import zmq

from whoc.interfaces.interface_base import InterfaceBase

# Code copied from ROSCO; consider just importing and using that code
# directly??


class ROSCO_ZMQInterface(InterfaceBase):
    def __init__(
        self, network_address="tcp://*:5555", identifier="0", timeout=600.0, verbose=False
    ):
        """Python implementation of the ZeroMQ server side for the ROSCO
        ZeroMQ wind farm control interface. This class makes it easy for
        users to receive measurements from ROSCO and then send back control
        setpoints (generator torque, nacelle heading and/or blade pitch
        angles).
        Args:
            network_address (str, optional): The network address to
                communicate over with the desired instance of ROSCO. Note that,
                if running a wind farm simulation in SOWFA or FAST.Farm, there
                are multiple instances of ROSCO and each of these instances
                needs to communicate over a unique port. Also, for each of those
                instances, you will need an instance of zmq_server. Defaults to
                "tcp://*:5555".
            identifier (str, optional): Turbine identifier. Defaults to "0".
            timeout (float, optional): Seconds to wait for a message from
                the ZeroMQ server before timing out. Defaults to 600.0.
            verbose (bool, optional): Print to console. Defaults to False.
        """
        super().__init__()

        self.network_address = network_address
        self.identifier = identifier
        self.timeout = timeout
        self.verbose = verbose
        self._connect()

    def _connect(self):
        """
        Connect to zmq server
        """
        address = self.network_address

        # Connect socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(address)

        if self.verbose:
            print("[%s] Successfully established connection with %s" % (self.identifier, address))

    def _disconnect(self):
        """
        Disconnect from zmq server
        """
        self.socket.close()
        context = zmq.Context()
        context.term()

    def get_measurements(self, _):
        """
        Receive measurements from ROSCO .dll
        """
        if self.verbose:
            print("[%s] Waiting to receive measurements from ROSCO..." % (self.identifier))

        # Initialize a poller for timeouts
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        timeout_ms = int(self.timeout * 1000)
        if poller.poll(timeout_ms):
            # Receive measurements over network protocol
            message_in = self.socket.recv_string()
        else:
            raise IOError(
                "[%s] Connection to '%s' timed out." % (self.identifier, self.network_address)
            )

        # Convert to individual strings and then to floats
        measurements = message_in
        measurements = measurements.replace("\x00", "").split(",")
        measurements = [float(m) for m in measurements]

        # Convert to a measurement dict
        measurements = dict(
            {
                "Turbine_ID": measurements[0],
                "iStatus": measurements[1],
                "Time": measurements[2],
                "VS_MechGenPwr": measurements[3],
                "VS_GenPwr": measurements[4],
                "GenSpeed": measurements[5],
                "RotSpeed": measurements[6],
                "GenTqMeas": measurements[7],
                "NacelleHeading": measurements[8],
                "NacelleVane": measurements[9],
                "HorWindV": measurements[10],
                "rootMOOP1": measurements[11],
                "rootMOOP2": measurements[12],
                "rootMOOP3": measurements[13],
                "FA_Acc": measurements[14],
                "NacIMU_FA_Acc": measurements[15],
                "Azimuth": measurements[16],
            }
        )

        if self.verbose:
            print("[%s] Measurements received:" % self.identifier, measurements)

        return measurements

    def check_controls(self, controls_dict):
        available_controls = [
            "turbine_ID",
            "genTorque",
            "nacelleHeading",
            "bladePitch",
        ]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration")

    def send_controls(
        self, turbine_ID=0, genTorque=0.0, nacelleHeading=0.0, bladePitch=[0.0, 0.0, 0.0]
    ):
        """
        Send controls to ROSCO .dll ffor individual turbine control

        Parameters:
        -----------
        genTorques: float
            Generator torque setpoint
        nacelleHeadings: float
            Nacelle heading setpoint
        bladePitchAngles: List (len=3)
            Blade pitch angle setpoint
        """
        # Create a message with controls to send to ROSCO
        message_out = b"%016.5f, %016.5f, %016.5f, %016.5f, %016.5f, %016.5f" % (
            turbine_ID,
            genTorque,
            nacelleHeading,
            bladePitch[0],
            bladePitch[1],
            bladePitch[2],
        )

        #  Send reply back to client
        if self.verbose:
            print("[%s] Sending setpoint string to ROSCO: %s." % (self.identifier, message_out))

        # Send control controls over network protocol
        self.socket.send(message_out)

        if self.verbose:
            print("[%s] Setpoints sent successfully." % self.identifier)

        return None

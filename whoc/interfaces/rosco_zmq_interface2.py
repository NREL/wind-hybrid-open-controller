import zmq
from rosco.toolbox.control_interface import wfc_zmq_server

from whoc.interfaces.interface_base import InterfaceBase


class ROSCO_ZMQInterface(InterfaceBase):
    def __init__(self, h_dict, zmq_dict):
        super().__init__()
        
        # Controller parameters
        if "controller" in h_dict and h_dict["controller"] is not None:
            self.controller_parameters = h_dict["controller"]
        else:
            self.controller_parameters = {}

        # Plant parameters
        if "plant" in h_dict and h_dict["plant"] is not None:
            self.plant_parameters = h_dict["plant"]
        else:
            self.plant_parameters = {}
       
        self.port = zmq_dict['port']
        self.network_address = f"tcp://*:{self.port}"
        self.timeout = zmq_dict['timeout']
        self.verbose = zmq_dict['verbose']
        self.logfile = zmq_dict['logfile']
        self.wfc_zmq_server = wfc_zmq_server(
            self.network_address, self.timeout, self.verbose, self.logfile
        )
        

    def get_measurements(self):
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

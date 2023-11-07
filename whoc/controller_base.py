import multiprocessing as mp
from zmq_server import whoc_zmq_server

from utilities import convert_absolute_nacelle_heading_to_offset

class ControllerBase():

    def __init__(self,
        use_zmq_interface=True,
        use_helics_interface=False,
        timeout=100.0,
        verbose=True
        ):

        if use_helics_interface:
            raise NotImplementedError(
                "HELICS interface has not yet been implemented."
            )

            # TODO: eventually, this would set up a federate (with same 
            # public methods as the whoc_zmq_server
            #self.s = whoc_helics_federate()
        
        elif use_zmq_interface:

            # TODO: set up HELICS server
            # Set up connections with each turbine
            self.s = whoc_zmq_server(network_address="tcp://*:5555",
                timeout=timeout, verbose=True)

        else:
            raise ValueError(
                "Must selecte either Zero-MQ or HELICS interface."
            )

    def receive_turbine_outputs(self):
        # May need to eventually loop here, depending on server set up.
        self.measurements_dict = self.s.get_measurements()

        return None

    def generate_turbine_references(self):
        # This function likely overridden by the child class.
        # Make this an abstract method?

        # Something very minimal here, based on ROSCO example 17.
        #west_offset = convert_absolute_nacelle_heading_to_offset(270,
        #    self.measurements_dict["NacelleHeading"])

        current_time = self.measurements_dict['Time']
        if current_time <= 10.0:
            yaw_setpoint = 0.0
        else:
            yaw_setpoint = 20.0

        self.offsets_to_send = {
            "turbine_ID":0, # TODO: hardcoded! Replace
            "genTorque":0.0,
            "nacelleHeading":west_offset,
            "bladePitch":[0.0, 0.0, 0.0]
        }

        return None

    def send_turbine_references(self):
        # May need to eventually loop here, depending on server set up.
        self.s.send_setpoints(
            turbine_ID=self.offsets_to_send["turbine_ID"],
            genTorque=self.offsets_to_send["genTorque"],
            nacelleHeading=self.offsets_to_send["nacelleHeading"],
            bladePitch=self.offsets_to_send["bladePitch"]
        )

        return None

    def run(self):

        connect_zmq = True
        while connect_zmq:
            self.receive_turbine_outputs()
            self.generate_turbine_references()
            self.send_turbine_references()

            if self.measurements_dict['iStatus'] == -1:
                connect_zmq = False
                self.s._disconnect()
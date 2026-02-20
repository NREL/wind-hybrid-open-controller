from hycon.controllers.controller_base import ControllerBase


class WakeSteeringROSCOStandin(ControllerBase):
    def __init__(self, interface):
        super.__init__(interface, timeout=100.0, verbose=True)

    def compute_controls(self):
        self.generate_turbine_references()

    def generate_turbine_references(self):
        # Something very minimal here, based on ROSCO example 17.
        # west_offset = convert_absolute_nacelle_heading_to_offset(270,
        #    self.measurements_dict["NacelleHeading"])

        current_time = self.measurements_dict["Time"]
        if current_time <= 10.0:
            yaw_setpoint = 0.0
        else:
            yaw_setpoint = 20.0

        self.controls_dict = {
            "turbine_ID": 0,  # TODO: hardcoded! Replace.
            "genTorque": 0.0,
            "nacelleHeading": yaw_setpoint,
            "bladePitch": [0.0, 0.0, 0.0],
        }

        return None

    # def run(self):

    #     connect_zmq = True
    #     while connect_zmq:
    #         self.receive_turbine_outputs()
    #         self.generate_turbine_references()
    #         self.send_turbine_references()

    #         if self.measurements_dict['iStatus'] == -1:
    #             connect_zmq = False
    #             self.s._disconnect()

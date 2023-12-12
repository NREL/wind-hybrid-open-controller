from whoc.controller_base import ControllerBase

class WakeSteeringADStandin(ControllerBase):

    def __init__(self, interface, input_dict):

        super().__init__(interface)

        self.dt = input_dict["dt"] # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)
        
        # Set initial conditions
        yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        if hasattr(yaw_IC, '__len__'):
            if len(yaw_IC) == self.n_turbines:
                self.setpoints_dict = {
                    "yaw_angles": yaw_IC
                }
            else:
                raise TypeError("yaw initial condition should be a float or "+\
                    "a list of floats of length num_turbines.")
        else:
            self.setpoints_dict = {
                "yaw_angles": [yaw_IC]*self.n_turbines
            }
        
        # Grab name of wind farm (assumes there is only one!)

    def compute_setpoints(self):
        self.generate_turbine_references()

    def generate_turbine_references(self):
        # Based on an early implementation for Hercules

        current_time = self.measurements_dict["time"]
        if current_time <= 10.0:
            yaw_setpoint = [270.0]*self.n_turbines
        else:
            yaw_setpoint = self.measurements_dict["wind_directions"]

        self.setpoints_dict = {
            "yaw_angles": yaw_setpoint
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
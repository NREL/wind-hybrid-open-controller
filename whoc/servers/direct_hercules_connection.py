# How will we handle other things here? May need to have a wind farm
# version, an electrolyzer version, etc...
from whoc.servers.server_base import ServerBase


class WHOC_AD_yaw_connection(ServerBase):

    def __init__(self, input_dict):

        super().__init__()

        self.dt = input_dict["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Grab name of wind farm (assumes there is only one!)
        self.wf_name = list(input_dict["hercules_comms"]["amr_wind"].keys())[0]

        pass

    def get_measurements(self, input_dict):

        wind_directions = input_dict["hercules_comms"]\
                                    ["amr_wind"]\
                                    [self.wf_name]\
                                    ["turbine_wind_directions"]
        wind_speeds = input_dict["hercules_comms"]\
                                ["amr_wind"]\
                                [self.wf_name]\
                                ["turbine_wind_speeds"]
        powers = input_dict["hercules_comms"]\
                           ["amr_wind"]\
                           [self.wf_name]\
                           ["turbine_powers"]

        measurements = {
            "wind_directions":wind_directions,
            "wind_speeds":wind_speeds,
            "turbine_powers":powers
        }

        return measurements

    def check_setpoints(self, setpoints_dict):
        
        available_setpoints = [
            "yaw_angles"
        ]
    
        for k in setpoints_dict.keys():
            if k not in available_setpoints:
                raise ValueError(
                    "Setpoint "+k+" is not available in this configuration"
                )
    
    def send_setpoints(self, input_dict, yaw_angles=None):

        if yaw_angles is None:
            yaw_angles = [0.]*self.n_turbines

        input_dict["hercules_comms"]\
                  ["amr_wind"]\
                  [self.wf_name]\
                  ["turbine_yaw_angles"] = yaw_angles

        return input_dict


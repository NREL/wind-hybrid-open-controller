import numpy as np


from whoc.controller_base import ControllerBase


class HerculesWindBatteryController(ControllerBase):
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

        self.dt = input_dict["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]

    def send_setpoints(self, hercules_dict):
        self._s.check_setpoints(self.setpoints_dict)
        dict = self._s.send_setpoints(hercules_dict, self.setpoints_dict)

        return dict  # or main_dict, or what?

    def step(self, hercules_dict=None):
        self.receive_measurements(hercules_dict)
        # receive measurements sets self.measurements_dict
        self.compute_setpoints()
        hercules_dict = self.send_setpoints(hercules_dict)

        return hercules_dict

    def compute_setpoints(self):
        # set self.setpoints_dict

        # calc wind setpoints
        wind_setpoints = self.calc_wind_setpoints()
        battery_setpoints = self.calc_battery_setpoints()

        self.setpoints_dict = {"wind": wind_setpoints, "battery": battery_setpoints}

        return None

    def calc_wind_setpoints(self):
        wind_setpoints = {}
        return wind_setpoints

    def calc_battery_setpoints(self):
        available_power = np.sum(self.measurements_dict["wind_farm"]["turbine_powers"])
        if available_power <= 1000:
            signal = available_power
        else:
            signal = -500

        battery_setpoints = {"signal": signal}
        return battery_setpoints

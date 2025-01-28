import numpy as np

from whoc.controllers.controller_base import ControllerBase


class WindBatteryController(ControllerBase):
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

        self.dt = input_dict["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]

    def send_controls(self, hercules_dict):
        self._s.check_controls(self.setpoints_dict)
        hercules_dict = self._s.send_controls(hercules_dict, self.setpoints_dict)

        return hercules_dict

    def step(self, hercules_dict=None):
        self._receive_measurements(hercules_dict)
        self.compute_controls()
        hercules_dict = self.send_controls(hercules_dict)

        return hercules_dict

    def compute_controls(self):
        wind_setpoints = self.calc_wind_setpoints()
        battery_setpoints = self.calc_battery_setpoints()

        self.setpoints_dict = {"wind": wind_setpoints, "battery": battery_setpoints}

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

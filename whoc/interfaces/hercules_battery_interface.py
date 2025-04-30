from whoc.interfaces.interface_base import InterfaceBase


class HerculesBatteryInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]

        # Grab name of battery (assumes there is only one! Takes the first)
        batteries_in_simulation = [k for k in hercules_dict["py_sims"] if "battery" in k]
        if len(batteries_in_simulation) == 0:
            raise ValueError("No battery found in simulation.")
        elif len(batteries_in_simulation) > 1:
            raise ValueError("Multiple batteries found in simulation. Only one is allowed.")
        else:
            self.battery_name = batteries_in_simulation[0]

    def get_measurements(self, hercules_dict):
        # Extract externally-provided power signal
        if ("external_signals" in hercules_dict
            and "plant_power_reference" in hercules_dict["external_signals"]):
            plant_power_reference = hercules_dict["external_signals"]["plant_power_reference"]
        else:
            plant_power_reference = 0

        measurements = {
            "time": hercules_dict["time"],
            "power_reference": plant_power_reference,
            "battery_power": -hercules_dict["py_sims"][self.battery_name]["outputs"]["power"],
            "battery_soc": hercules_dict["py_sims"][self.battery_name]["outputs"]["soc"]
        }

        return measurements

    def check_controls(self, controls_dict):
        available_controls = ["power_setpoint"]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def send_controls(self, hercules_dict, power_setpoint=0):

        hercules_dict["py_sims"]["inputs"].update({"battery_signal": -power_setpoint})

        return hercules_dict

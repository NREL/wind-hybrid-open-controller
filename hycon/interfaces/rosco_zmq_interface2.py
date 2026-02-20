from hycon.interfaces.interface_base import InterfaceBase


class ROSCO_ZMQInterface(InterfaceBase):
    def __init__(self, h_dict):
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

        # Emulator parameters
        if "emulator" in h_dict and h_dict["emulator"] is not None:
            self.emulator_parameters = h_dict["emulator"]
        else:
            self.emulator_parameters = {}

    def get_measurements(self):
        measurements = self.measurements

        return measurements

    def check_controls(self):
        pass
    
    def update_setpoints(self, id, current_time, measurements):
        self.measurements = measurements
        if current_time <= 10.0:
            YawOffset = 0.0
        else:
            if id == 1:
                YawOffset = DESIRED_YAW_OFFSET[0]
            else:
                YawOffset = DESIRED_YAW_OFFSET[1]


        setpoints = {}
        setpoints["ZMQ_YawOffset"] = YawOffset
        return setpoints

    def send_controls(
        self, turbine_ID=0, genTorque=0.0, nacelleHeading=0.0, bladePitch=[0.0, 0.0, 0.0]
    ):
        pass

from hycon.interfaces.interface_base import InterfaceBase


class Hycon_python_server(InterfaceBase):
    def __init__():
        pass

    def get_measurements(self, dict):
        # Possibly need to extract the measurements from the input dict?
        measurements = dict

        return measurements

    def send_controls(self, dict):
        # Not sure if anything needs to be done here. Can possibly return None.
        controls = dict

        return controls

from abc import ABCMeta, abstractmethod


class ControllerBase(metaclass=ABCMeta):
    def __init__(self, interface, verbose = True):
        self._s = interface
        self.verbose = verbose

        # Initialize measurements and controls to send
        self._measurements_dict = {}
        self._controls_dict = {}

    def _receive_measurements(self, input_dict=None):
        # May need to eventually loop here, depending on server set up.
        self._measurements_dict = self._s.get_measurements(input_dict)

        return None

    def _send_controls(self, input_dict=None):
        self._s.check_controls(self._controls_dict)
        output_dict = self._s.send_controls(input_dict, **self._controls_dict)

        return output_dict

    def step(self, input_dict=None):
        # If not running with direct hercules integration, hercules_dict may simply be None
        # throughout this method.
        self._receive_measurements(input_dict)

        self._controls_dict = self.compute_controls(self._measurements_dict)

        output_dict = self._send_controls(input_dict)

        return output_dict

    @abstractmethod
    def compute_controls(self, measurements_dict: dict) -> dict:
        pass  # Control algorithms should be implemented in the compute_controls 
        # method of the child class. 

from abc import ABCMeta, abstractmethod


class EstimatorBase(metaclass=ABCMeta):
    def __init__(self, interface, verbose = True):
        self._s = interface
        self.verbose = verbose

        self._measurements_dict = {}
        self._estimates_dict = {}

    def _receive_measurements(self, input_dict=None):
        self._measurements_dict = self._s.get_measurements(input_dict)
        return None

    def _send_estimates(self, input_dict=None):
        self._s.check_estimates(self._estimates_dict)
        output_dict = self._s.send_estimates(input_dict, **self._controls_dict)

        return output_dict

    def step(self, input_dict=None):
        self._receive_measurements(input_dict)

        output_dict = self.compute_estimates(self._measurements_dict)

        return output_dict

    @abstractmethod
    def compute_estimates(self, measurements_dict: dict) -> dict:
        raise NotImplementedError(
            "compute_estimates method must be implemented in the child class of EstimatorBase."
        )
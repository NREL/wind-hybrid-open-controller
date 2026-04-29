import copy
import inspect
from abc import ABCMeta, abstractmethod


class ControllerBase(metaclass=ABCMeta):
    def __init__(self, interface, cname=None, verbose=True):
        self._s = interface
        self.verbose = verbose

        self.cname = cname

        # Initialize measurements and controls to send
        self._measurements_dict = {}
        self._controls_dict = {}

    def _receive_measurements(self, input_dict=None):
        # May need to eventually loop here, depending on server set up.
        self._measurements_dict = self._s.get_measurements(input_dict)

        return None

    def _send_controls(self, input_dict=None):
        self._s.check_controls(self._controls_dict)
        output_dict = self._s.send_controls(input_dict, self._controls_dict)

        return output_dict

    def step(self, input_dict=None):
        # If not running with direct hercules integration, hercules_dict may simply be None
        # throughout this method.
        self._receive_measurements(input_dict)

        self._controls_dict = self.compute_controls(self._measurements_dict)

        output_dict = self._send_controls(input_dict)

        return output_dict

    def check_controller_parameters(self, controller_parameters):
        # Check valid controller parameters
        valid_controller_parameters = inspect.getfullargspec(self.set_controller_parameters).args
        valid_controller_parameters.remove("self")
        invalid_cps = [
            cp for cp in controller_parameters.keys() if cp not in valid_controller_parameters
        ]
        if len(invalid_cps) > 0:
            raise KeyError(
                "Found keys "
                + str(invalid_cps)
                + " in controller_parameters, but they are not valid controller parameters for "
                + self.__class__.__name__
                + ". Valid controller parameters are: "
                + str(valid_controller_parameters)
                + "."
            )

        # Check that required parameters are specified
        default_values = inspect.getfullargspec(self.set_controller_parameters).defaults
        num_defaults = len(default_values) if default_values is not None else 0

        required_args = valid_controller_parameters[:-num_defaults]

        missing_required_cps = set(required_args) - set(controller_parameters.keys())
        if len(missing_required_cps) > 0:
            raise KeyError("Missing required controller parameters: " + str(missing_required_cps))

        return None
    
    # TODO: Consider an "update controller parameters" method. Not urgent.

    def compute_controls_without_updating_state(self, measurements_dict):
        """
        Compute controls without updating internal state. This is used when the control output
        needs to be queried without actually updating the controller's internal state, such as
        in the hybrid supervisory controller when querying component controllers for their desired
        power references without actually updating their states.
        """
        self._initial_state = copy.deepcopy(self.__dict__)
        controls_dict = self.compute_controls(measurements_dict)
        self.__dict__.update(copy.deepcopy(self._initial_state))
        return controls_dict

    @abstractmethod
    def set_controller_parameters(self, **kwargs):
        raise NotImplementedError("set_controller_parameters must be implemented in child class.")

    @property
    def controller_parameters(self):
        return self._s.controller_parameters

    @property
    def plant_parameters(self):
        return self._s.plant_parameters

    @property
    def dt(self):
        return self._s.dt

    @dt.setter
    def dt(self, _):
        print(
            "Warning: Setting dt directly is deprecated. Use the interface's dt property instead."
        )

    @property
    def cname(self):
        if hasattr(self, "_cname"):
            if self._cname is None:
                raise ValueError("cname has been set to None for this controller.")
            else:
                return self._cname
        else:
            return ValueError("cname has not been set for this controller.")

    @cname.setter
    def cname(self, value):
        if not isinstance(value, (str, type(None))):
            raise ValueError("cname must be a string.")
        self._cname = value

    @abstractmethod
    def compute_controls(self, measurements_dict: dict) -> dict:
        pass  # Control algorithms should be implemented in the compute_controls
        # method of the child class.

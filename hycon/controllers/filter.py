import numpy as np
from scipy import signal

from hycon.controllers.controller_base import ControllerBase


class LowPassFilter(ControllerBase):
    """
    Applies a low-pass filter to the reference trajectory, returning a filtered
    version.
    """

    def __init__(self, interface, cname, controller_parameters=None, verbose=True):
        """
        Instantiates the low-pass filter.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the corresponding
                plant component.
            controller_parameters (dict): Dictionary of filter polynomial coefficients, with keys
            'a' and 'b' for the denominator and numerator coefficients, respectively. Assumes
            specified for a continuous-time filter, which will be discretized.
            verbose (bool): If True, print debug information. 'a' and 'b' should be specified as 1D
            numpy arrays, and the length of a determines the order of the filter.
        """

        # Handle None controller_parameters
        super().__init__(interface, cname, verbose)

        # Initialize filter coefficients and state
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(
        self,
        a,
        b,
    ):
        """
        Sets the filter coefficients.

        Args:
            a (list): Denominator coefficients of the filter polynomial.
            b (list): Numerator coefficients of the filter polynomial.
        """
        # Convert to numpy arrays if not already
        a = np.asarray(a)
        b = np.asarray(b)

        # Check DC gain of the filter is 1, raise warning if not
        if a[-1] != b[-1]:
            print(
                "Warning: DC gain of the low-pass filter is not 1. "
                "This will lead to a steady-state scaling of the reference trajectory."
            )

        # Convert to discrete-time filter coefficients using bilinear transform
        self._bz, self._az, _ = signal.cont2discrete((b, a), self.dt, method="bilinear")
        self._bz = self._bz.ravel()

        # Initialize filter state
        self._x = np.zeros(len(self._az) - 1)

    def compute_controls(self, measurements_dict):
        """
        Computes the control action by applying the low-pass filter to the reference trajectory.
        """

        u = measurements_dict[self.cname]["power_reference"]
        y, self._x = signal.lfilter(self._bz, self._az, [u], zi=self._x)

        return {self.cname: {"power_setpoint": float(y[0])}}

import numpy as np
from scipy import signal


class LowPassFilter:
    """
    Generate a low-pass filter that can be applied to a signal.
    """

    def __init__(self, a, b, dt):
        """
        Instantiates the low-pass filter.

        Args:
            a (list | np.array): Denominator coefficients of the filter polynomial.
            b (list | np.array): Numerator coefficients of the filter polynomial.
            dt (float): Time step for discretization of the filter.
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
        self._bz, self._az, _ = signal.cont2discrete((b, a), dt, method="bilinear")
        self._bz = self._bz.ravel()

        # Initialize filter state
        self._x = np.zeros(len(self._az) - 1)

    def __call__(self, u):
        """
        Computes the filter output for a given (scalar) input
        """

        y, self._x = signal.lfilter(self._bz, self._az, [u], zi=self._x)

        return y[0]


class RateLimiter:
    def __init__(self, max_rate_up, max_rate_down=None, dt=1.0):
        """
        Instantiates the rate limiter.

        Args:
            max_rate_up (float): Maximum allowed rate of increase of the signal (units per second).
            max_rate_down (float): Maximum allowed rate of decrease of the signal (units per
                second).
            dt (float): Time step for discretization of the filter.
        """

        self._max_rate_up = max_rate_up * dt
        self._max_rate_down = -(max_rate_down or max_rate_up) * dt

        self._x = 0.0

    def __call__(self, u):
        """
        Computes the rate-limited output for a given (scalar) input.
        """

        delta = u - self._x
        delta_clipped = np.clip(delta, self._max_rate_down, self._max_rate_up)
        output = self._x + delta_clipped
        self._x = output

        return output


class Saturator:
    def __init__(self, min_value=None, max_value=None):
        """
        Instantiates the saturator.

        Args:
            min_value (float): Minimum allowed value of the signal.
            max_value (float): Maximum allowed value of the signal.
        """

        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, u):
        """
        Computes the saturated output for a given (scalar) input.
        """

        return np.clip(u, self._min_value, self._max_value)

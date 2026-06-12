import numpy as np
from hycon.controllers.signal_conditioning import LowPassFilter


def test_LowPassFilter():
    """
    Tests that the LowPassFilter outputs a reasonable signal
    """
    dt_test = 1.0

    # First-order filter
    lpf = LowPassFilter(a=[1, 0.25], b=[0.25], dt=dt_test)

    # Test with step input
    power_setpoint_ref = 1000
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf(power_setpoint_ref)
    assert np.all(np.diff(out) > 0)  # Check output is increasing
    assert np.all(out < power_setpoint_ref)  # Check output is below reference

    # Confirm steady-state behavior
    for _ in range(100):
        y = lpf(power_setpoint_ref)
    assert np.isclose(y, power_setpoint_ref, atol=1e-2)

    # Second-order overdamped filter
    omega_n = 0.5
    zeta = 1.5
    lpf = LowPassFilter(a=[1, 2 * zeta * omega_n, omega_n**2], b=[omega_n**2], dt=dt_test)

    # Test with step input
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf(power_setpoint_ref)
    assert np.all(np.diff(out) > 0)  # Check output is increasing
    assert np.all(out < power_setpoint_ref)  # Check output is below reference

    # Second-order underdamped filter
    zeta = 0.5
    lpf = LowPassFilter(a=[1, 2 * zeta * omega_n, omega_n**2], b=[omega_n**2], dt=dt_test)

    # Test with step input
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf(power_setpoint_ref)
    # Check underdamped (oscillates, overshoots)
    assert np.any(np.diff(out) < 0)
    assert np.any(out > power_setpoint_ref)

    # Steady-state behavior should still hold
    for _ in range(100):
        y = lpf(power_setpoint_ref)
    assert np.isclose(y, power_setpoint_ref, atol=1e-2)

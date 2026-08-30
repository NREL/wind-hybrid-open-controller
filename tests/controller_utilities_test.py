import numpy as np
from hycon.controllers.signal_conditioning import LowPassFilter, RateLimiter, Saturator


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


def test_RateLimiter():
    """
    Tests that the RateLimiter outputs a reasonable signal
    """
    dt_test = 2.0
    max_rate_up = 100
    max_rate_down = 50
    rate_limiter = RateLimiter(max_rate_up=max_rate_up, max_rate_down=max_rate_down, dt=dt_test)

    # Test with input
    rate_limiter._x = 500
    out = rate_limiter(1000)
    assert out == 500 + 2 * 100
    out = rate_limiter(-1000)
    assert out == 500 + 2 * 100 - 2 * 50

    # Test not specifying max_rate_down (uses max_rate_up for both)
    rate_limiter = RateLimiter(max_rate_up=max_rate_up, dt=dt_test)
    rate_limiter._x = 500
    out = rate_limiter(-1000)
    assert out == 500 - 2 * 100


def test_Saturator():
    """
    Tests that the Saturator outputs a reasonable signal
    """
    saturator = Saturator(min_value=0, max_value=100)

    # Test with input
    out = saturator(50)
    assert out == 50
    out = saturator(-10)
    assert out == 0
    out = saturator(150)
    assert out == 100

    # Test with only max_value
    saturator = Saturator(max_value=100)
    out = saturator(150)
    assert out == 100
    out = saturator(-10)
    assert out == -10

    # Test with only min_value
    saturator = Saturator(min_value=0)
    out = saturator(-10)
    assert out == 0
    out = saturator(150)
    assert out == 150

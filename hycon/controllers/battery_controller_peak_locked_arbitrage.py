"""MISO-specific peak-locked battery controller with price-responsive off-peak arbitrage."""

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from hycon.controllers.controller_base import ControllerBase

# Peak blocks are not necessarily daily: a morning peak on one day followed by
# an evening peak on the next can be well over 24 hours apart. A gap beyond this
# bound, however, indicates the load peak forecast is missing data rather than a
# legitimately spaced peak, so it is treated as a fault.
MAX_TIME_TO_NEXT_PEAK_BLOCK = pd.Timedelta(48, "h")

# RT market interval; the RT offer price covers replacing one interval of rated discharge.
RT_INTERVAL = pd.Timedelta(5, "min")

_ONE_HOUR = pd.Timedelta(1, "h")


class BatteryControllerPeakLockedArbitrage(ControllerBase):
    """Peak-locked battery controller with price-responsive off-peak arbitrage.

    The battery follows a four-hour peak block schedule, where it never charges.
    Outside peak blocks, it may charge, discharge, or idle on each step. In and
    out of peak, discharge requires the real-time price to clear the RT offer
    price: the marginal day-ahead price of replacing the current deficit plus
    one RT interval of rated discharge, divided by round-trip efficiency, plus
    ``discharge_margin``. In peak, replacement is planned over the charging
    window following the current block. Off peak, RT replacement excludes the
    current hour, whose DA price is stale once RT is visibly elevated, so there
    is no off-peak RT discharge in the final hour before a peak. Off peak, the
    battery charges when the day-ahead price is at or below the marginal price
    of the current deficit, and must charge once the remaining time equals the
    rated-power time needed to reach full charge before the next peak block.

    The charge plan reuses expected base-plant surplus and the cheapest
    day-ahead hours before the next peak. Grid charging preserves headroom for
    forecast excess power; actual excess is absorbed regardless of price.

    Besides ``power_setpoint``, each step returns ``charge_price``,
    ``rt_offer_price`` and ``da_offer_price`` ($/MWh), and 0/1 ``rt_available``
    and ``in_peak`` flags, so they can be logged as Hercules channels. The DA
    offer price is planned once per peak block for a full block of discharge
    and is NaN off peak.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Instantiate BatteryControllerPeakLockedArbitrage.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the
                corresponding plant component.
            controller_parameters (dict): Dictionary of controller parameters. See
                set_controller_parameters for the supported keys.
            verbose (bool): If True, print debug information.
        """
        super().__init__(interface, cname, verbose)

        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

        self.rated_power_charging = self.plant_parameters[self.cname]["charge_rate"]
        self.rated_power_discharging = self.plant_parameters[self.cname]["discharge_rate"]
        self.min_SOC = self.plant_parameters[self.cname]["state_of_charge_min"]
        self.max_SOC = self.plant_parameters[self.cname]["state_of_charge_max"]
        self.roundtrip_efficiency = self.plant_parameters[self.cname]["roundtrip_efficiency"]
        self.energy_capacity = self.plant_parameters[self.cname]["energy_capacity"]

        # Terminal energy to charge from empty to full; energy_capacity is
        # deliverable, so all round-trip losses apply.
        self.charge_energy_capacity = self.energy_capacity / self.roundtrip_efficiency

        # Empty-to-full charge time at rated charge power (hours).
        self.charge_duration = self.charge_energy_capacity / self.rated_power_charging

        # Set before the load peak forecast is validated so spacing checks can
        # reject overlapping blocks.
        self.peak_block_duration = pd.Timedelta(4, "h")

        # Terminal energy to recharge one RT interval / one peak block of rated discharge.
        self.rt_replacement_energy = (
            self.rated_power_discharging * (RT_INTERVAL / _ONE_HOUR) / self.roundtrip_efficiency
        )
        self.block_discharge_energy = self.rated_power_discharging * (
            self.peak_block_duration / _ONE_HOUR
        )

        self.initialize_excess_power_forecast()
        self.initialize_load_peak_forecast()

        # Peak timing and the charging-window plan are refreshed hourly.
        self.prev_hour = None
        self.prev_peak_block_start = None
        self.next_peak_block_start = None
        self._plan = self._rt_plan = (np.empty(0), np.empty(0), 0.0)

        # The DA offer price is planned once per peak block and held across it.
        self._da_offer_block_start = None
        self._da_offer_price = np.nan

    def set_controller_parameters(
        self,
        load_peak_forecast_filename,
        discharge_margin,
        excess_power_forecast_filename=None,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set parameters for BatteryControllerPeakLockedArbitrage.

        Args:
            load_peak_forecast_filename (str): Path to a feather file containing
                the expected forecasted load peak. Must contain a column ``time_utc``
                where each entry marks the start of a 4-hour peak block. Blocks
                need not be spaced exactly 24 hours apart, but must be
                hour-aligned and at least the block duration apart.
            discharge_margin (float): Margin ($/MWh) the real-time LMP must
                exceed above the round-trip-efficiency-adjusted marginal charge
                price before discharging, in and out of peak. Typically set to
                the battery's degradation cost per MWh discharged.
            excess_power_forecast_filename (str): Path to a feather file
                containing the expected base-plant excess power (power above the
                interconnect limit). Must contain columns ``time_utc`` and
                ``excess_power`` (in kW, non-negative). If not provided, expected
                excess is assumed to be zero (not co-located with the battery).
        """

        # check that load_peak_forecast_filename is a string
        if not isinstance(load_peak_forecast_filename, str):
            raise ValueError("load_peak_forecast_filename must be a string")

        # Check the file exists
        if not Path(load_peak_forecast_filename).is_file():
            raise ValueError(f"File {load_peak_forecast_filename} does not exist")

        # Save the load peak forecast filename
        self.load_peak_forecast_filename = load_peak_forecast_filename

        # Check that the discharge_margin is a float
        if not isinstance(discharge_margin, Real):
            raise ValueError("discharge_margin must be a float")

        # Save the discharge margin
        self.discharge_margin = float(discharge_margin)

        # Check that excess_power_forecast_filename is a string or None
        if excess_power_forecast_filename is not None:
            if not isinstance(excess_power_forecast_filename, str):
                raise ValueError("excess_power_forecast_filename must be a string or None")
            if not Path(excess_power_forecast_filename).is_file():
                raise ValueError(f"File {excess_power_forecast_filename} does not exist")

        # Save the excess power forecast filename
        self.excess_power_forecast_filename = excess_power_forecast_filename

    def initialize_excess_power_forecast(self):
        """
        Initialize the excess power forecast.

        Builds two views of the base plant's expected excess power (power above
        the interconnect limit, already computed by the file's producer):
        ``excess_power`` at the forecast's native resolution for per-step
        ``.asof()`` lookups, and ``excess_energy`` aggregated to hourly energy
        for charge planning. Both are ``None`` when no excess forecast is
        configured.
        """
        # If no forecast is provided, treat expected excess as zero.
        if self.excess_power_forecast_filename is None:
            self.excess_power = None
            self.excess_energy = None
            return

        df = pd.read_feather(self.excess_power_forecast_filename)[["time_utc", "excess_power"]]

        # Save excess power as a Series indexed by time_utc for fast .asof() lookups
        self.excess_power = df.set_index("time_utc")["excess_power"].sort_index()

        # Native time step of the forecast, used to convert power to energy.
        excess_power_dt = df["time_utc"].diff().mode()[0]

        # excess_power is in kW; aggregate to hourly kWh for charge planning.
        df["excess_energy"] = df["excess_power"] * excess_power_dt.total_seconds() / 3600.0
        self.excess_energy = (
            df.set_index("time_utc")["excess_energy"].resample("h").sum().sort_index()
        )

    def initialize_load_peak_forecast(self):
        """
        Initialize the load peak forecast.

        Reads the peak block start times and validates that they form a clean,
        hour-aligned schedule with no overlapping blocks. Blocks may be spaced
        arbitrarily (not necessarily daily). The hour alignment is relied on by
        the per-step transition logic, so it is enforced here at construction
        rather than discovered mid-simulation.
        """
        # Load load peak
        df_load_peak = pd.read_feather(self.load_peak_forecast_filename)

        if "time_utc" not in df_load_peak.columns:
            raise ValueError("load peak forecast must contain a 'time_utc' column")

        if not pd.api.types.is_datetime64_any_dtype(df_load_peak["time_utc"]):
            raise ValueError("load peak forecast 'time_utc' must be datetime")

        # Each entry marks the start of a 4-hour peak block. Sort
        # chronologically so peak block lookups can rely on ordering.
        peak_block_starts = pd.DatetimeIndex(df_load_peak["time_utc"]).sort_values()

        if len(peak_block_starts) == 0:
            raise ValueError("load peak forecast is empty")

        # Peak blocks must start on the hour; the per-step transition logic
        # refreshes block timing only hourly and assumes hour-aligned starts.
        if not (peak_block_starts == peak_block_starts.floor("h")).all():
            raise ValueError("load peak forecast block starts must be hour-aligned")

        # Peaks may be spaced arbitrarily (not necessarily daily), but
        # consecutive starts closer than the block duration would overlap, which
        # also catches duplicate entries.
        if len(peak_block_starts) > 1:
            spacings = peak_block_starts[1:] - peak_block_starts[:-1]
            if (spacings < self.peak_block_duration).any():
                raise ValueError(
                    "load peak forecast block starts must be at least the peak block duration apart"
                )

        self.peak_block_starts = peak_block_starts

    def get_peak_block_timing(self, time_utc):
        """Locate the peak block starts bracketing a point in time.

        Finds the most recent peak block starting at or before ``time_utc`` and
        the next peak block starting strictly after it. Peak blocks need not be
        daily, so the next block may be more than 24 hours away; a gap beyond
        ``MAX_TIME_TO_NEXT_PEAK_BLOCK`` instead indicates the load peak forecast
        is missing data.

        A next peak block must always exist. The previous peak block start may be
        ``None`` (e.g. when ``time_utc`` precedes the first block in the
        forecast).

        Args:
            time_utc (pd.Timestamp): The time at which to evaluate the peak
                block timing.

        Returns:
            tuple: A 2-tuple of ``(prev_peak_block_start, next_peak_block_start)``
                where each entry is a ``pd.Timestamp``. ``prev_peak_block_start``
                is ``None`` when no block precedes ``time_utc``;
                ``next_peak_block_start`` is always populated.

        Raises:
            ValueError: If no peak block follows ``time_utc``, or if the next
                peak block is more than ``MAX_TIME_TO_NEXT_PEAK_BLOCK`` away.
        """
        peak_block_starts = self.peak_block_starts

        # Block starts are sorted, so a binary search locates the split point in
        # O(log n): entries before ``split`` are <= time_utc, the rest are after.
        split = peak_block_starts.searchsorted(time_utc, side="right")

        # Most recent peak block at or before time_utc (None if time_utc precedes all blocks)
        if split > 0:
            prev_peak_block_start = peak_block_starts[split - 1]
        else:
            prev_peak_block_start = None

        # Next peak block strictly after time_utc. A next block must always
        # exist; a missing one indicates the load peak forecast is exhausted.
        if split >= len(peak_block_starts):
            raise ValueError("time_utc has no upcoming peak block")
        next_peak_block_start = peak_block_starts[split]

        # Peaks need not be daily, but a gap beyond MAX_TIME_TO_NEXT_PEAK_BLOCK
        # signals missing forecast data rather than legitimate spacing.
        if next_peak_block_start - time_utc > MAX_TIME_TO_NEXT_PEAK_BLOCK:
            raise ValueError(
                f"time_utc is more than {MAX_TIME_TO_NEXT_PEAK_BLOCK} from the next peak block"
            )

        return prev_peak_block_start, next_peak_block_start

    def _extend_day_ahead_lmps(self, day_ahead_lmps, n_hours):
        """Trim or pad a day-ahead LMP array to cover a charging window.

        The measured day-ahead array is a fixed 24-hour horizon starting at the
        current hour. When peaks are spaced more than 24 hours apart the charging
        window can be longer than that horizon, so the array is extended by
        repeating its final hour. Padded hours tie with the last real hour, and
        because charge planning sorts with a stable argsort they always rank
        after it, keeping real and earlier hours preferred.

        Args:
            day_ahead_lmps (array-like): The measured day-ahead LMPs starting at
                the current hour.
            n_hours (int): Number of hours the charging window spans.

        Returns:
            numpy.ndarray: Day-ahead LMPs of length ``n_hours``, trimmed when the
                window is shorter than the input and padded with the last value
                when it is longer.
        """
        day_ahead_lmps = np.asarray(day_ahead_lmps, dtype=float)
        if n_hours <= day_ahead_lmps.size:
            return day_ahead_lmps[:n_hours]
        padding = np.full(n_hours - day_ahead_lmps.size, day_ahead_lmps[-1])
        return np.concatenate([day_ahead_lmps, padding])

    def _charge_energy_to_full(self, soc):
        """Terminal energy needed to charge from ``soc`` to ``max_SOC``.

        Uses ``charge_energy_capacity`` so the returned energy is measured at the
        battery terminals, matching the base-plant surplus and rated charge
        power the charge plan compares it against.

        Args:
            soc (float): Current battery state of charge.

        Returns:
            float: Terminal energy (kWh) required to reach ``max_SOC``, clipped
                at zero when the battery is already at or above ``max_SOC``.
        """
        return max(0.0, self.max_SOC - soc) * self.charge_energy_capacity

    def _charge_time_to_full(self, soc):
        """Time to charge from ``soc`` to ``max_SOC`` at rated charge power.

        The battery must stop discretionary discharge once the next peak block
        is this close, because a partly charged battery needs proportionally
        less time to fill than an empty one.

        Args:
            soc (float): Current battery state of charge.

        Returns:
            pd.Timedelta: Time required to reach ``max_SOC`` from ``soc`` at
                rated charge power.
        """
        return pd.Timedelta(max(0.0, self.max_SOC - soc) * self.charge_duration, "h")

    def _plan_window(self, current_hour, day_ahead_lmps_24, window_start, window_end):
        """Plan the hourly charging window ``[window_start, window_end)``.

        Stores ``_plan`` over the full window and ``_rt_plan`` over the window
        without the current hour (identical in peak, where the window starts
        after the block). ``_marginal_price`` prices energy needs against these
        plans per step.

        Args:
            current_hour (pd.Timestamp): The current hour (floored).
            day_ahead_lmps_24 (array-like): The next 24 hourly day-ahead LMPs
                starting at ``current_hour``.
            window_start (pd.Timestamp): Hour-aligned window start, at or after
                ``current_hour`` (the end of the current block when in peak).
            window_end (pd.Timestamp): Start of the next peak block.
        """
        hours = pd.date_range(window_start, window_end - _ONE_HOUR, freq="h")
        if self.excess_energy is not None:
            excess = self.excess_energy.reindex(hours, fill_value=0.0).to_numpy()
        else:
            excess = np.zeros(len(hours))

        offset = int((window_start - current_hour) / _ONE_HOUR)
        prices = self._extend_day_ahead_lmps(
            np.asarray(day_ahead_lmps_24, dtype=float)[offset:], len(hours)
        )
        self._plan = self._sorted_plan(prices, excess)
        skip = 1 if window_start == current_hour else 0
        self._rt_plan = self._sorted_plan(prices[skip:], excess[skip:])

    def _sorted_plan(self, prices, excess):
        """Sort hourly prices cheapest first with cumulative purchasable grid energy.

        Args:
            prices (numpy.ndarray): Hourly day-ahead LMPs ($/MWh).
            excess (numpy.ndarray): Hourly expected excess energy (kWh).

        Returns:
            tuple: ``(sorted_prices, cumulative_grid_energy, excess_energy)``, where
                purchasable energy per hour is rated charge power less that hour's
                expected excess, and ``excess_energy`` is the total absorbable
                expected excess (kWh).
        """
        cheapest_first = np.argsort(prices, kind="stable")
        purchasable = np.maximum(self.rated_power_charging - excess, 0.0)
        # Excess above rated is curtailed anyway, so it is not credited.
        excess_energy = float(np.minimum(excess, self.rated_power_charging).sum())
        return prices[cheapest_first], np.cumsum(purchasable[cheapest_first]), excess_energy

    def _marginal_price(self, charge_energy, plan):
        """Marginal day-ahead price ($/MWh) of supplying ``charge_energy`` over a plan.

        Expected excess is credited first, then the cheapest hours are bought
        until the remainder is covered.

        Args:
            charge_energy (float): Terminal energy (kWh) to charge.
            plan (tuple): Plan from ``_sorted_plan``.

        Returns:
            float: Price of the marginal hour. ``0.0`` when expected excess
                covers the need or the plan is empty (back-to-back blocks);
                the plan's max price plus 1 when it cannot supply the need.
        """
        sorted_prices, cumulative_grid_energy, excess_energy = plan
        energy_from_grid = charge_energy - excess_energy
        if energy_from_grid <= 0 or sorted_prices.size == 0:
            return 0.0
        idx = np.searchsorted(cumulative_grid_energy, energy_from_grid)
        if idx < sorted_prices.size:
            return float(sorted_prices[idx])
        return float(sorted_prices[-1]) + 1.0

    def _offer_price(self, charge_energy, plan):
        """Price ($/MWh) at which discharging pays off, given ``charge_energy`` to replace."""
        marginal_price = self._marginal_price(charge_energy, plan)
        return marginal_price / self.roundtrip_efficiency + self.discharge_margin

    def compute_controls(self, measurements_dict):
        """
        Compute the battery power setpoint and offer prices for the current step.

        Peak timing and the charging-window plan refresh on the first step of
        each hour (peak blocks are hour-aligned); prices are re-evaluated every
        step from the live SOC.

        Args:
            measurements_dict (dict): Measurements from the simulator. Expected
                keys: ``time_utc``, ``RT_LMP``, ``DA_LMP``, ``DA_LMP_24hours``,
                and ``self.cname`` (with at least ``state_of_charge``).

        Returns:
            dict: ``{self.cname: {...}}`` with ``power_setpoint`` (kW, charging
                negative), ``charge_price``, ``rt_offer_price``,
                ``da_offer_price`` ($/MWh), and 0/1 ``rt_available`` and
                ``in_peak``.
        """
        time_utc = measurements_dict["time_utc"]
        current_hour = time_utc.floor("h")
        soc = measurements_dict[self.cname]["state_of_charge"]
        real_time_lmp = measurements_dict["RT_LMP"]
        day_ahead_lmp = measurements_dict["DA_LMP"]

        new_hour = self.prev_hour is None or current_hour > self.prev_hour
        if new_hour:
            self.prev_peak_block_start, self.next_peak_block_start = self.get_peak_block_timing(
                time_utc
            )
        in_peak = (
            self.prev_peak_block_start is not None
            and time_utc - self.prev_peak_block_start < self.peak_block_duration
        )
        if new_hour:
            # In peak, replacement is planned over the window after this block.
            window_start = (
                self.prev_peak_block_start + self.peak_block_duration if in_peak else current_hour
            )
            self._plan_window(
                current_hour,
                measurements_dict["DA_LMP_24hours"],
                window_start,
                self.next_peak_block_start,
            )
        self.prev_hour = current_hour

        deficit = self._charge_energy_to_full(soc)
        charge_price = self._marginal_price(deficit, self._plan)
        rt_offer_price = self._offer_price(deficit + self.rt_replacement_energy, self._rt_plan)

        if not in_peak:
            self._da_offer_price = np.nan
        elif self._da_offer_block_start != self.prev_peak_block_start:
            usable_energy = max(0.0, soc - self.min_SOC) * self.energy_capacity
            block_replacement_energy = (
                min(self.block_discharge_energy, usable_energy) / self.roundtrip_efficiency
            )
            self._da_offer_price = self._offer_price(deficit + block_replacement_energy, self._plan)
            self._da_offer_block_start = self.prev_peak_block_start

        # Off peak, discretionary discharge stops early enough to be full at the next peak.
        must_charge = not in_peak and (
            self.next_peak_block_start - time_utc <= self._charge_time_to_full(soc)
        )
        rt_available = in_peak or not (
            must_charge or soc <= self.min_SOC or self._rt_plan[0].size == 0
        )

        if rt_available and real_time_lmp >= rt_offer_price:
            power_setpoint = self.rated_power_discharging
        elif in_peak:
            power_setpoint = 0.0
        else:
            # Grid buys leave SOC headroom for expected excess still to come.
            grid_room = deficit - self._plan[2]
            if (must_charge or day_ahead_lmp <= charge_price) and grid_room > 0:
                power_setpoint = -self.rated_power_charging
            else:
                power_setpoint = 0.0

            if self.excess_power is not None:
                excess = self.excess_power.asof(time_utc)
                if not pd.isna(excess) and excess > 0:
                    # Absorb all current excess, up to the charge rating.
                    power_setpoint = max(
                        -self.rated_power_charging, min(power_setpoint, -float(excess))
                    )

        if (power_setpoint > 0 and soc <= self.min_SOC) or (
            power_setpoint < 0 and soc >= self.max_SOC
        ):
            power_setpoint = 0.0

        # Apply limitations based on super controller
        power_limit_lower = measurements_dict[self.cname].get("power_limit_lower", -np.inf)
        power_limit_upper = measurements_dict[self.cname].get("power_limit_upper", np.inf)
        power_setpoint = np.clip(power_setpoint, power_limit_lower, power_limit_upper)

        return {
            self.cname: {
                "power_setpoint": power_setpoint,
                "charge_price": charge_price,
                "rt_offer_price": rt_offer_price,
                "da_offer_price": self._da_offer_price,
                "rt_available": int(rt_available),
                "in_peak": int(in_peak),
            }
        }

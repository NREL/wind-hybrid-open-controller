# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from time import time

from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface

def simulate_controller(controller_class, input_dict, **kwargs):
    print(f"Running instance of {controller_class.__name__}")
    # Load a FLORIS object for AEP calculations
    fi = ControlledFlorisInterface(yaw_limits=input_dict["controller"]["yaw_limits"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
    
    ctrl = controller_class(fi, input_dict=input_dict, **kwargs)
    # TODO use coroutines or threading for hercules interfaces
    # optionally warm-start with LUT solution
    
    yaw_angles_ts = []
    yaw_angles_change_ts = []
    turbine_powers_ts = []

    convergence_time_ts = []
    # opt_codes_ts = []

    opt_cost_ts = []
    opt_cost_terms_ts = []
    
    fi.reset(disturbances={"wind_speeds": [kwargs["wind_mag_ts"][0]],
                            "wind_directions": [kwargs["wind_dir_ts"][0]]},
                            # "turbulence_intensities": [wind_ti_ts[0]]},
                            init_controls_dict={"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines})
    
    for k, t in enumerate(np.arange(0, kwargs["episode_max_time"] - input_dict["dt"], input_dict["dt"])):

        # feed interface with new disturbances
        fi.step(disturbances={"wind_speeds": [kwargs["wind_mag_ts"][k]],
                                "wind_directions": [kwargs["wind_dir_ts"][k]]},
                                seed=2)
                                # "turbulence_intensities": [wind_ti_ts[k]]})
        # fi_mpc.env.floris.flow_field.u, fi_mpc.env.floris.flow_field.u
        # receive measurements from interface, compute control actions, and send to interface
        ctrl.current_freestream_measurements = [
                    kwargs["wind_mag_ts"][k] * np.cos((270. - kwargs["wind_dir_ts"][k]) * (np.pi / 180.)),
                    kwargs["wind_mag_ts"][k] * np.sin((270. - kwargs["wind_dir_ts"][k]) * (np.pi / 180.))
                ]
        start_time = time()
        ctrl.step()
        end_time = time()
        convergence_time_ts.append(end_time - start_time)
        # opt_codes_ts.append(ctrl.opt_code)
        if hasattr(ctrl, "opt_cost"):
            opt_cost_terms_ts.append(ctrl.opt_cost_terms)
            opt_cost_ts.append(ctrl.opt_cost)
        else:
            opt_cost_terms_ts.append([np.nan] * 2)
            opt_cost_ts.append(np.nan)
        
        if hasattr(ctrl, "init_sol"):
            init_states = np.array(ctrl.init_sol['states']) * ctrl.yaw_norm_const
            init_ctrl_inputs = ctrl.init_sol['control_inputs']
        else:
            init_states = [np.nan] * ctrl.n_turbines
            init_ctrl_inputs = [np.nan] * ctrl.n_turbines
        
        yaw_angles_ts.append(ctrl.controls_dict['yaw_angles'])
        # yaw_angles_change_ts.append(ctrl.opt_sol['control_inputs'][:ctrl.n_turbines])
        yaw_angles_change_ts.append(yaw_angles_ts[-1] - (yaw_angles_ts[-2] if k > 0 else ctrl.yaw_IC))
        turbine_powers_ts.append(ctrl.measurements_dict['powers'])

        # print(f"Time = {ctrl.measurements_dict['time']} for Optimizer {MPC.optimizers[optimizer_idx].__class__}")
        print(f"\nTime = {ctrl.measurements_dict['time']}",
            f"Measured Freestream Wind Direction = {kwargs['wind_dir_ts'][k]}",
            f"Measured Freestream Wind Magnitude = {kwargs['wind_mag_ts'][k]}",
            f"Measured Turbine Wind Directions = {ctrl.measurements_dict['wind_directions']}",
            f"Measured Turbine Wind Magnitudes = {ctrl.measurements_dict['wind_speeds']}",
            f"Measured Yaw Angles = {ctrl.measurements_dict['yaw_angles']}",
            f"Measured Turbine Powers = {ctrl.measurements_dict['powers']}",
            f"Initial Yaw Angle Solution = {init_states}",
            f"Initial Yaw Angle Change Solution = {init_ctrl_inputs}",
            # f"Optimizer Output = {ctrl.opt_code['text']}",
            # f"Optimized Yaw Angle Solution = {ctrl.opt_sol['states'] * ctrl.yaw_norm_const}",
            # f"Optimized Yaw Angle Change Solution = {ctrl.opt_sol['control_inputs']}",
            f"Optimized Yaw Angles = {yaw_angles_ts[-1]}",
            f"Optimized Yaw Angle Changes = {yaw_angles_change_ts[-1]}",
            f"Optimized Power Cost = {opt_cost_terms_ts[-1][0]}",
            f"Optimized Yaw Change Cost = {opt_cost_terms_ts[-1][1]}",
            f"Convergence Time = {convergence_time_ts[-1]}",
            sep='\n')
        


        # send yaw angles to compute lut solution
        # fi_lut.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
        #                     "wind_directions": [wind_dir_ts[k]]},
        #                     seed=2)
                            # "turbulence_intensities": [wind_ti_ts[k]]})
        
        # receive measurements from interface, compute control actions, and send to interface
        # ensure feasibility for lut and greedy approaches
        # ctrl_lut.step()
        # lut_yaw_angles = np.clip(ctrl_lut.controls_dict["yaw_angles"], lut_yaw_angles - ctrl_lut.dt * ctrl_lut.yaw_rate, lut_yaw_angles + ctrl_lut.dt * ctrl_lut.yaw_rate)
        # ctrl.fi.env.calculate_wake((ctrl_lut.measurements_dict["wind_directions"] - lut_yaw_angles)[np.newaxis, :])
        # lut_yawed_turbine_powers = np.squeeze(ctrl.fi.env.get_turbine_powers())

        # greedy_yaw_angles = np.clip(ctrl_lut.measurements_dict["wind_directions"], greedy_yaw_angles - ctrl_lut.dt * ctrl_lut.yaw_rate, greedy_yaw_angles + ctrl_lut.dt * ctrl_lut.yaw_rate)
        # ctrl.fi.env.calculate_wake((ctrl_lut.measurements_dict["wind_directions"] - greedy_yaw_angles)[np.newaxis, :])
        # greedy_yaw_turbine_powers = np.squeeze(ctrl.fi.env.get_turbine_powers())

        # if k > 0:
        #     print(f"Change in Optimized Farm Powers relative to previous time-step = {100 * (sum(turbine_powers_ts[-1]) - sum(turbine_powers_ts[-2])) / sum(turbine_powers_ts[-2])} %")
        #     print(f"Change in Optimized Cost relative to previous time-step = {100 * (opt_cost_ts[-1] - opt_cost_ts[-2]) /  abs(opt_cost_ts[-2])} %")
        #     print(f"Change in Optimized Cost relative to first time-step = {100 * (opt_cost_ts[-1] - opt_cost_ts[0]) /  abs(opt_cost_ts[0])} %")
        #     # print(f"Change in Optimized Cost relative to second time-step = {100 * (opt_costs_ts[-1] - opt_costs_ts[1]) /  opt_costs_ts[1]} %")

        #     print(f"Change in Optimized Farm Powers relative to Greedy Yaw = {100 * (sum(turbine_powers_ts[-1]) - sum(greedy_yaw_turbine_powers)) /  sum(greedy_yaw_turbine_powers)} %")
        #     print(f"Change in Optimized Farm Powers relative to LUT = {100 * (sum(turbine_powers_ts[-1]) - sum(lut_yawed_turbine_powers)) /  sum(lut_yawed_turbine_powers)} %")

    yaw_angles_ts = np.vstack(yaw_angles_ts)
    yaw_angles_change_ts = np.vstack(yaw_angles_change_ts)
    turbine_powers_ts = np.vstack(turbine_powers_ts)
    opt_cost_terms_ts = np.vstack(opt_cost_terms_ts)

    results_df = pd.DataFrame(data={
        **{
            f"TurbineYawAngle_{i}": yaw_angles_ts[:, i] for i in range(input_dict["controller"]["num_turbines"])
        }, 
        **{
            f"TurbineYawAngleChange_{i}": yaw_angles_change_ts[:, i] for i in range(input_dict["controller"]["num_turbines"])
        },
        **{
            f"TurbinePower_{i}": turbine_powers_ts[:, i] for i in range(input_dict["controller"]["num_turbines"])
        },
        "FarmPower": np.sum(turbine_powers_ts, axis=1),
        **{
            f"OptimizationCostTerm_{i}": opt_cost_terms_ts[:, i] for i in range(opt_cost_terms_ts.shape[1])
        },
        "TotalOptimizationCost": np.sum(opt_cost_terms_ts, axis=1),
        "OptimizationConvergenceTime": convergence_time_ts
    })

    return results_df

class ControllerBase(metaclass=ABCMeta):
    def __init__(self, interface, verbose=True):
        self._s = interface
        self.verbose = verbose

        # if use_helics_interface:
        #     raise NotImplementedError(
        #         "HELICS interface has not yet been implemented."
        #     )

        #     # TODO: eventually, this would set up a federate (with same
        #     # public methods as the whoc_zmq_server
        #     #self._s = whoc_helics_federate()

        # elif use_zmq_interface:
        #     from servers.zmq_server import WHOC_zmq_server

        #     # TODO: set up HELICS server
        #     # Set up connections with each turbine
        #     self._s = WHOC_zmq_server(network_address="tcp://*:5555",
        #         timeout=timeout, verbose=True)

        # elif use_direct_hercules_connection:
        #     from servers.direct_hercules_connection import WHOC_AD_yaw_connection
        #     self._s = WHOC_AD_yaw_connection(hercules_dict)

        # else:
        #     from servers.python_server import WHOC_python_server
        #     self._s = WHOC_python_server()

        # Initialize controls to send
        self.controls_dict = None

    def _receive_measurements(self, hercules_dict=None):
        # May need to eventually loop here, depending on server set up.
        self.measurements_dict = self._s.get_measurements(hercules_dict)

        return None

    def _send_controls(self, hercules_dict=None) -> dict:
        
        self._s.check_controls(self.controls_dict)
        controller_output = self._s.send_controls(hercules_dict, **self.controls_dict)

        return controller_output  # or main_dict, or what?

    def step(self, hercules_dict=None):
        # If not running with direct hercules integration,
        # hercules_dict may simply be None throughout this method.
        self._receive_measurements(hercules_dict)

        self.compute_controls() # set self.controls_dict

        hercules_dict = self._send_controls(hercules_dict)

        return hercules_dict  # May simply be None.

    @abstractmethod
    def compute_controls(self):
        # Control algorithms should be implemented in the compute_controls
        # method of the child class.
        pass

from whoc.interfaces import (
    HerculesADYawInterface,
    HerculesWindBatteryInterface,
    ROSCO_ZMQInterface
)

test_input_dict ={
    "dt": 1,
    "controller":{"num_turbines": 2},
    "hercules_comms":{"amr_wind": {"test_farm":{}}},
}


def test_interface_instantiation():

    _ = HerculesADYawInterface(input_dict=test_input_dict)
    _ = HerculesWindBatteryInterface(input_dict=test_input_dict)
    _ = ROSCO_ZMQInterface()
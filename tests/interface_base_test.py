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

import pytest

from whoc.interfaces.interface_base import InterfaceBase

class InheritanceTestClassBad(InterfaceBase):
    """
    Class that is missing necessary methods.
    """
    def __init__(self):
        super().__init__()

class InheritanceTestClassGood(InterfaceBase):
    """
    Class that is missing necessary methods.
    """
    def __init__(self):
        super().__init__()

    def get_measurements(self):
        pass

    def check_setpoints(self):
        pass

    def send_controls(self):
        pass

def test_InterfaceBase_methods():
    """
    Check that the base interface class establishes the correct methods.
    """
    interface_base = InheritanceTestClassGood()
    assert hasattr(interface_base, "get_measurements")
    assert hasattr(interface_base, "check_setpoints")
    assert hasattr(interface_base, "send_controls")

def test_inherited_methods():
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """

    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad()

    _ = InheritanceTestClassGood()

def test_all_interfaces_implement_methods():
    
    # TODO: import and instantiate all interface classes.
    pass


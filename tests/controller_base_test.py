import pytest
from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.interface_base import InterfaceBase


class StandinInterface(InterfaceBase):
    """
    Empty class to test controllers.
    """

    def __init__(self):
        super().__init__()

    def get_measurements(self):
        pass

    def check_controls(self):
        pass

    def send_controls(self):
        pass


class InheritanceTestClassBad(ControllerBase):
    """
    Class that is missing necessary methods.
    """

    def __init__(self, interface):
        super().__init__(interface)


class InheritanceTestClassGood(ControllerBase):
    """
    Class that is missing necessary methods.
    """

    def __init__(self, interface):
        super().__init__(interface)

    def compute_controls(self):
        pass


def test_ControllerBase_methods():
    """
    Check that the base interface class establishes the correct methods.
    """
    test_interface = StandinInterface()

    controller_base = InheritanceTestClassGood(test_interface)
    assert hasattr(controller_base, "_receive_measurements")
    assert hasattr(controller_base, "_send_controls")
    assert hasattr(controller_base, "step")
    assert hasattr(controller_base, "compute_controls")


def test_inherited_methods():
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """
    test_interface = StandinInterface()

    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad(test_interface)

    _ = InheritanceTestClassGood(test_interface)

import pytest
from hycon.controllers.controller_base import ControllerBase


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


def test_ControllerBase_methods(test_interface_standin):
    """
    Check that the base interface class establishes the correct methods.
    """
    controller_base = InheritanceTestClassGood(test_interface_standin)
    assert hasattr(controller_base, "_receive_measurements")
    assert hasattr(controller_base, "_send_controls")
    assert hasattr(controller_base, "step")
    assert hasattr(controller_base, "compute_controls")


def test_inherited_methods(test_interface_standin):
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """
    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad(test_interface_standin)

    _ = InheritanceTestClassGood(test_interface_standin)

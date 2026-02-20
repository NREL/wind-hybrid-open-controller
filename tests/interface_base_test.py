import pytest
from hycon.interfaces.interface_base import InterfaceBase

# import hycon.interfaces


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

    def check_controls(self):
        pass

    def send_controls(self):
        pass


def test_InterfaceBase_methods():
    """
    Check that the base interface class establishes the correct methods.
    """
    interface_base = InheritanceTestClassGood()
    assert hasattr(interface_base, "get_measurements")
    assert hasattr(interface_base, "check_controls")
    assert hasattr(interface_base, "send_controls")


def test_inherited_methods():
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """

    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad()

    _ = InheritanceTestClassGood()


def test_all_interfaces_implement_methods():
    # In future, I'd like to dynamically instantiate classes, but the different
    # inputs that they require on __init__ is currently a roadblock, so I'll just
    # explicitly instantiate each interface class for the time being.

    # class_dict = dict(inspect.getmembers(hycon.interfaces, inspect.isclass))

    pass

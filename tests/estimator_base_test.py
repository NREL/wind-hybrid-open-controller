import pytest
from whoc.estimators.estimator_base import EstimatorBase
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


class InheritanceTestClassBad(EstimatorBase):
    """
    Class that is missing necessary methods (compute_estimates).
    """

    def __init__(self, interface):
        super().__init__(interface)


class InheritanceTestClassGood(EstimatorBase):
    """
    Class that is missing necessary methods.
    """

    def __init__(self, interface):
        super().__init__(interface)

    def compute_estimates(self):
        pass


def test_EstimatorBase_methods():
    """
    Check that the base interface class establishes the correct methods.
    """
    test_interface = StandinInterface()

    estimator_base = InheritanceTestClassGood(test_interface)
    assert hasattr(estimator_base, "_receive_measurements")
    # assert hasattr(estimator_base, "_send_estimates") # Not yet sure we want this.
    assert hasattr(estimator_base, "step")
    assert hasattr(estimator_base, "compute_estimates")


def test_inherited_methods():
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """
    test_interface = StandinInterface()

    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad(test_interface)

    _ = InheritanceTestClassGood(test_interface)

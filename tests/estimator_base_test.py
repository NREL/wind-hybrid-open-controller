import pytest
from hycon.estimators.estimator_base import EstimatorBase


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


def test_EstimatorBase_methods(test_interface_standin):
    """
    Check that the base interface class establishes the correct methods.
    """
    estimator_base = InheritanceTestClassGood(test_interface_standin)
    assert hasattr(estimator_base, "_receive_measurements")
    # assert hasattr(estimator_base, "_send_estimates") # Not yet sure we want this.
    assert hasattr(estimator_base, "step")
    assert hasattr(estimator_base, "compute_estimates")


def test_inherited_methods(test_interface_standin):
    """
    Check that a subclass of InterfaceBase inherits methods correctly.
    """
    with pytest.raises(TypeError):
        _ = InheritanceTestClassBad(test_interface_standin)

    _ = InheritanceTestClassGood(test_interface_standin)

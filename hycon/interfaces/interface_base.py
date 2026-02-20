from abc import ABCMeta, abstractmethod


class InterfaceBase(metaclass=ABCMeta):
    def __init__(self):
        self._dt = None
        self._plant_parameters = None
        self._controller_parameters = None

    @abstractmethod
    def get_measurements(self):
        raise NotImplementedError

    @abstractmethod
    def check_controls(self):
        raise NotImplementedError

    @abstractmethod
    def send_controls(self):
        raise NotImplementedError

    @property
    def dt(self):
        if self._dt is None:
            raise AttributeError(
                f"{self.__class__.__name__} does not have 'dt' defined. ",
            )
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def plant_parameters(self):
        if self._plant_parameters is None:
            raise AttributeError(
                f"{self.__class__.__name__} does not have 'plant_parameters' defined. ",
            )
        return self._plant_parameters

    @plant_parameters.setter
    def plant_parameters(self, value):
        self._plant_parameters = value

    @property
    def controller_parameters(self):
        if self._controller_parameters is None:
            raise AttributeError(
                f"{self.__class__.__name__} does not have 'controller_parameters' defined. ",
            )
        return self._controller_parameters

    @controller_parameters.setter
    def controller_parameters(self, value):
        self._controller_parameters = value

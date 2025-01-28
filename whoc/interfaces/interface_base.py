from abc import ABCMeta, abstractmethod


class InterfaceBase(metaclass=ABCMeta):
    @abstractmethod
    def get_measurements(self):
        raise NotImplementedError

    @abstractmethod
    def check_controls(self):
        raise NotImplementedError

    @abstractmethod
    def send_controls(self):
        raise NotImplementedError

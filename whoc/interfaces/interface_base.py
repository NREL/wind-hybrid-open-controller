from abc import abstractmethod

class InterfaceBase():

    @abstractmethod
    def get_measurements(self):
        pass
    
    @abstractmethod
    def check_setpoints(self):
        pass
    
    @abstractmethod
    def send_setpoints(self):
        pass

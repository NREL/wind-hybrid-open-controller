from abc import abstractmethod
import multiprocessing as mp
from servers.zmq_server import WHOC_zmq_server
from servers.python_server import WHOC_python_server

from utilities import convert_absolute_nacelle_heading_to_offset

class ControllerBase():

    def __init__(self,
        use_zmq_interface=False,
        use_helics_interface=False,
        timeout=100.0,
        verbose=True
        ):

        if use_helics_interface:
            raise NotImplementedError(
                "HELICS interface has not yet been implemented."
            )

            # TODO: eventually, this would set up a federate (with same 
            # public methods as the whoc_zmq_server
            #self._s = whoc_helics_federate()
        
        elif use_zmq_interface:

            # TODO: set up HELICS server
            # Set up connections with each turbine
            self._s = WHOC_zmq_server(network_address="tcp://*:5555",
                timeout=timeout, verbose=True)

        else:
            self._s = WHOC_python_server()

        # Initialize setpoints to send
        self.setpoints_dict = None

    def receive_measurements(self, dict=None):
        # May need to eventually loop here, depending on server set up.
        self.measurements_dict = self._s.get_measurements(dict)

        return None

    def send_setpoints(self):

        self._s.check_setpoints(self.setpoints_dict)
        self._s.send_setpoints(**self.setpoints_dict)

        return self.setpoints_dict # or main_dict, or what?

    def step(self, dict=None):

        self.receive_measurements(dict)

        self.compute_setpoints()

        self.send_setpoints()

        return 
    
    @abstractmethod
    def compute_setpoints(self):
        # Control algorithms should be implemented in the compute_setpoints
        # method of the child class.
        pass

    # def run(self):

    #     connect_zmq = True
    #     while connect_zmq:
    #         self.receive_turbine_outputs()
    #         self.generate_turbine_references()
    #         self.send_turbine_references()

    #         if self.measurements_dict['iStatus'] == -1:
    #             connect_zmq = False
    #             self.s._disconnect()
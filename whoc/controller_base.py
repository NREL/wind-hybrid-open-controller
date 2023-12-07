from abc import abstractmethod


class ControllerBase():

    def __init__(self,
        interface,
        hercules_dict=None,
        timeout=100.0,
        verbose=True
        ):

        self._s = interface

        # if use_helics_interface:
        #     raise NotImplementedError(
        #         "HELICS interface has not yet been implemented."
        #     )

        #     # TODO: eventually, this would set up a federate (with same 
        #     # public methods as the whoc_zmq_server
        #     #self._s = whoc_helics_federate()
        
        # elif use_zmq_interface:
        #     from servers.zmq_server import WHOC_zmq_server

        #     # TODO: set up HELICS server
        #     # Set up connections with each turbine
        #     self._s = WHOC_zmq_server(network_address="tcp://*:5555",
        #         timeout=timeout, verbose=True)

        # elif use_direct_hercules_connection:
        #     from servers.direct_hercules_connection import WHOC_AD_yaw_connection
        #     self._s = WHOC_AD_yaw_connection(hercules_dict)

        # else:
        #     from servers.python_server import WHOC_python_server
        #     self._s = WHOC_python_server()

        # Initialize setpoints to send
        self.setpoints_dict = None

    def receive_measurements(self, dict=None):
        # May need to eventually loop here, depending on server set up.
        self.measurements_dict = self._s.get_measurements(dict)

        return None

    def send_setpoints(self, dict=None):

        self._s.check_setpoints(self.setpoints_dict)
        dict = self._s.send_setpoints(dict, **self.setpoints_dict)

        return dict # or main_dict, or what?

    def step(self, hercules_dict=None):

        # If not running with direct hercules integration, 
        # hercules_dict may simply be None throughout this method.
        self.receive_measurements(hercules_dict)

        self.compute_setpoints()

        hercules_dict = self.send_setpoints(hercules_dict)

        return hercules_dict # May simply be None.
    
    @abstractmethod
    def compute_setpoints(self):
        # Control algorithms should be implemented in the compute_setpoints
        # method of the child class.
        pass
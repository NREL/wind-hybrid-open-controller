import multiprocessing as mp
import subprocess

# from rosco.toolbox.control_interface import wfc_zmq_connections, wfc_zmq_server
from openfast_toolbox.fastfarm import ROSCOControllerInterface


class ROSCO_ZMQEmulator:
    def __init__(self, controller,input_dict):
        self.controller = controller
        self.interface = controller._s
        self.wfc_zmq_server = self.interface.wfc_zmq_server
        self.input_dict = input_dict

    def run_zmq(self):
        """Start the ZeroMQ server for wind farm control"""
        self.wfc_zmq_server.runserver()

    def run_FF(self):
        fstffile = self.input_dict['fstffile']
        ROSCOControllerInterface(fstffile)
        subprocess.run(["FAST.Farm",fstffile])

    def run(self):
        """Start the ZeroMQ server for wind farm control"""
        p_server = mp.Process(target=self.run_zmq)
        p_ff = mp.Process(target=self.run_FF)

        p_server.start()
        p_ff.start()

        p_server.join()
        p_ff.join()

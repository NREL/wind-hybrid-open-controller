import multiprocessing as mp
import subprocess

# from rosco.toolbox.control_interface import wfc_zmq_connections, wfc_zmq_server
from openfast_toolbox.fastfarm.ROSCOControllerInterface import ROSCOControllerInterface
from rosco.toolbox.control_interface import wfc_zmq_server

# mp.set_start_method('spawn', force=True)


def wfc_controller(id, current_time, measurements):
    DESIRED_YAW_OFFSET = [10, -10]
    if current_time <= 10.0:
        YawOffset = 0.0
    else:
        if id == 1:
            YawOffset = DESIRED_YAW_OFFSET[0]
        else:
            YawOffset = DESIRED_YAW_OFFSET[1]

    setpoints = {}
    setpoints["ZMQ_YawOffset"] = YawOffset
    return setpoints


def startzmqserver(port, timeout, verbose, logfile):
    """Start the ZeroMQ server for wind farm control"""
    zmqserver = wfc_zmq_server(f"tcp://*:{port}", timeout, verbose, logfile)
    zmqserver.wfc_controller = wfc_controller
    zmqserver.runserver()


def run_sim(emulatorinstance):
    simexe = emulatorinstance.interface.emulator_parameters["simexe"]
    siminput = emulatorinstance.interface.emulator_parameters["siminput"]
    if emulatorinstance.interface.emulator_parameters["simtype"] == "FAST.Farm":
        ROSCOControllerInterface(
            siminput, port=emulatorinstance.interface.emulator_parameters["port"]
        )
    else:
        raise NotImplementedError("Only FAST.Farm is supported as a simtype currently")
    subprocess.run([simexe, siminput], check=True)


class ROSCO_ZMQEmulator:
    def __init__(self, controller, input_dict):
        self.controller = controller
        self.interface = controller._s
        self.input_dict = input_dict

    def run(self):
        """Start the ZeroMQ server for wind farm control"""
        port = self.interface.emulator_parameters["port"]
        timeout = self.interface.emulator_parameters["timeout"]
        verbose = self.interface.emulator_parameters["verbose"]
        logfile = self.interface.emulator_parameters["logfile"]

        p_server = mp.Process(target=startzmqserver, args=(port, timeout, verbose, logfile))
        p_sim = mp.Process(target=run_sim, args=(self,))

        p_server.start()
        p_sim.start()

        p_server.join()
        p_sim.join()
    def formattedcontroller(self,id, current_time, measurements):
        pass
        

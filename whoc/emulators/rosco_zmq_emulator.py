import multiprocessing as mp
import subprocess
import os

# from rosco.toolbox.control_interface import wfc_zmq_connections, wfc_zmq_server
from openfast_toolbox.fastfarm.ROSCOControllerInterface import ROSCOControllerInterface
from rosco.toolbox.control_interface import wfc_zmq_server

mp.set_start_method('spawn', force=True)

PORT = 5559
def run_FF_free():
    ffexe = "/home/abhineet/MySoftware/miniforge3/envs/MySci/bin/FAST.Farm"
    fstffile = '/home/abhineet/Work/FFInterface/AG_WHOC/examples/FAST.Farm_interface/inputs/FFFiles/Case0_wdirp00/Seed_0/FFarm_mod.fstf'
    ROSCOControllerInterface(fstffile,port = PORT)
    # subprocess.run(["FAST.Farm",fstffile])
    # os.system(f"{ffexe} {fstffile} >> /dev/null")
    subprocess.run([ffexe,fstffile], check=True)
def run_zmq_free():
    """Start the ZeroMQ server for wind farm control"""
    print("########### Starting surver ###############")
    wfc_zmq_server_free = wfc_zmq_server(
        "tcp://*:{PORT}", 60, True, 'log.txt'
    )
    wfc_zmq_server_free.runserver()

class ROSCO_ZMQEmulator:
    def __init__(self, controller,input_dict):
        self.controller = controller
        self.interface = controller._s
        self.wfc_zmq_server = self.interface.wfc_zmq_server
        self.input_dict = input_dict

    def run_zmq(self):
        """Start the ZeroMQ server for wind farm control"""
        print("########### Starting surver ###############")
        self.wfc_zmq_server.runserver()

    def run_FF(self):
        ffexe = "/home/abhineet/MySoftware/miniforge3/envs/MySci/bin/FAST.Farm"
        fstffile = self.input_dict['plant']['fstffile']
        ROSCOControllerInterface(fstffile,port = self.interface.port)
        # subprocess.run(["FAST.Farm",fstffile])
        os.system(f"{ffexe} {fstffile} >> /dev/null")
        # subprocess.run([ffexe,fstffile], check=True)

    def run(self):
        """Start the ZeroMQ server for wind farm control"""
        p_server = mp.Process(target=run_zmq_free)
        # p_ff = mp.Process(target=self.run_FF)
        p_ff = mp.Process(target=run_FF_free)

        p_server.start()
        p_ff.start()
        
        p_server.join()
        p_ff.join()


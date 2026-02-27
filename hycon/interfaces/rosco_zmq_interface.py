import multiprocessing as mp
import subprocess

from rosco.toolbox.control_interface import wfc_zmq_server

from hycon.interfaces.interface_base import InterfaceBase


class ROSCO_ZMQInterface(InterfaceBase):
    def __init__(self, h_dict):
        super().__init__()

        # Controller parameters
        if "controller" in h_dict and h_dict["controller"] is not None:
            self.controller_parameters = h_dict["controller"]
        else:
            self.controller_parameters = {}

        # Plant parameters
        if "plant" in h_dict and h_dict["plant"] is not None:
            self.plant_parameters = h_dict["plant"]
        else:
            self.plant_parameters = {}

        # Emulator parameters
        if "emulator" in h_dict and h_dict["emulator"] is not None:
            self.emulator_parameters = h_dict["emulator"]
        else:
            self.emulator_parameters = {}

    def get_measurements(self):
        pass

    def check_controls(self):
        pass

    # def addcontroller(self, controller):
    #     self.controller = controller

    def send_controls(
        self, turbine_ID=0, genTorque=0.0, nacelleHeading=0.0, bladePitch=[0.0, 0.0, 0.0]
    ):
        pass


class ROSCO_Emulator():
    def __init__(self,interface,controller):
        self.interface = interface
        self.controller = controller

    def startserverandsim(self):
        pserver = mp.Process(target=self.run_zmq, args=())
        psim = mp.Process(target=self.rumfarmsim, args=())

        pserver.start()
        psim.start()

        psim.join()
        pserver.join()

    def run_zmq(self):
        """Start the ZeroMQ server for wind farm control"""
        # Start the server at the following address

        network_address = f"tcp://*:{self.interface.emulator_parameters['port']}"
        server = wfc_zmq_server(network_address, timeout=60.0, verbose=False, logfile="log.txt")

        # Provide the wind farm control algorithm as the wfc_controller method of the server
        i_wfc_cont = intermediate_wfc_controller(self.interface,self.controller)
        server.wfc_controller = i_wfc_cont

        # Run the server to receive measurements and send setpoints
        server.runserver()
    
    def rumfarmsim(self):
        simexe = self.interface.emulator_parameters["simexe"]
        siminput = self.interface.emulator_parameters["siminput"]
        simcmd = f"{simexe} {siminput}"
        print(f"Running simulation with command '{simcmd}'")
        subprocess.run(simcmd, shell=True, check=True)

class intermediate_wfc_controller:
    def __init__(self,interface,controller):
        self.interface = interface
        self.controller = controller
        self.n_turbines = self.interface.plant_parameters["n_turbines"]
        self.measurements_to_hycon_controller = {k:0 for k in range(self.n_turbines)}
        self.controls_from_hycon_controller = {}
    
    def update_setpoints(self, id, current_time, measurements):
        if len(self.measurements_to_hycon_controller) == self.n_turbines:
            self.controls_from_hycon_controller = self.getcontrolsfromhycon()
            self.measurements_to_hycon_controller = {}
        
        self.measurements_to_hycon_controller[id] = measurements['NacVane']
        
        setpoints = {}
        setpoints["ZMQ_YawOffset"] = self.controls_from_hycon_controller[id]
        return setpoints

    def getcontrolsfromhycon(self):
        measurements_dict = {}
        measurements_dict["wind_farm"] = {}
        measurements_dict["wind_farm"]["wind_directions"] = [
            self.measurements_to_hycon_controller[i] for i in range(self.n_turbines)
        ]
        self.controller.compute_controls(measurements_dict)



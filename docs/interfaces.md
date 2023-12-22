# Interfaces

The `whoc.interfaces` module contains a library of interfaces for connecting 
WHOC with various simulation platforms and other repositories. Each controller
run will require an `interface`, which is an instantiated object of a class
in this library. All interface classes should inherit from `InterfaceBase`, 
which can be found n interface_base.py, and should implement three methods:
- `get_measurements()`: Recieve measurements from simulation assets and 
organize into a dictionary that the calling controller can utilize. Optionally,
receives a large dictionary (for example, the Hercules `main_dict`), from which
useable measurements can be extracted/repackaged for easy use in the controller.
- `check_controls()`: Check that the keys in `controls_dict` are viable for 
the receiving plant.
- `send_controls()`: Send controls to the simulation assets. Controls are 
created as specific keyword arguements, which match those controls generated
by the calling controller. Optionally, receives a large dictionary 
(for example, the Hercules `main_dict`), which can be written to and returned
with controls as needed.

These methods will all be called in the `step()` method of `ControllerBase`.

## Available interfaces

### HerculesADYawInterface
For direct python communication with Hercules. This should be instantiated 
in a runscript that is running Hercules; used to generate a `controller` from 
the WHOC controllers submodule; and that `controller` should be passed to the
Hercules `Emulator` upon its instantiation.

### ROSCO_ZMQInterface
For sending and receiving communications from one or more ROSCO instances 
(which are likely connected to OpenFAST and FAST.Farm). Uses ZeroMQ to pass
messages between workers.

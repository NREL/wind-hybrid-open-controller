# Code development
To contribute to WHOC, please consider forking the main github repository,
with the [NREL repo](https://github.com/NREL/wind-hybrid-open-controller) as an 
upstream remote. See the [Installation instructions](install_instructions) 
for details about how to set up your repository as a developer.

To submit a new feature or bug fix, create a new branch 
in your fork and submit a pull request back to the `develop` branch in the 
main repo. The pull request will be reviewed by other WHOC maintainers and 
merged (using "squash and merge") into the `develop` branch. Periodically, 
the `develop` branch will be merged into the `main` branch and a version 
number will be assigned.

Unless an existing controller or interface exist to suit your needs, most 
users will need to generate:
- A new interface class inheriting from `InterfaceBase`
- A new controller class, implementing the desired control algorithm and 
inheriting from `ControllerBase`

Additionally, if you'd like to contribute to this base repository, please 
include in your pull request:
- Unit tests for the implemented controller
- Possibly unit tests for the implemented interface, if needed
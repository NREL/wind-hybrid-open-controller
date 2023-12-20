# Installation

WHOC is _not_ designed to be used as a stand-alone package. Most likely, 
you'll want to add WHOC to an existing conda environment that contains your
simulation testbed, such as [Hercules](https://github.com/NREL/hercules). 
For example, see the [Hercules installation instuctions](\
https://nrel.github.io/hercules/install_instructions.html) for how to set up
an appropriate conda environment.

## General users

If you intend to use WHOC, but not contribute, the following lines should
be sufficient to install WHOC (presumably, after activating your conda 
environment):

```
git clone https://github.com/NREL/wind-hybrid-open-controller
pip install -e wind-hybrid-open-controller
```

## Developers

If you intend to contribute to WHOC, we request that your fork the WHOC 
repository on github. You can then install WHOC (again, assuming you have 
already activated your conda environment) according to:

```
git clone https://github.com/your-github-id/wind-hybrid-open-controller



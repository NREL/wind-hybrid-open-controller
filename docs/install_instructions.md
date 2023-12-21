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
pip install wind-hybrid-open-controller/
```

## Developers

If you intend to contribute to WHOC, we request that your fork the WHOC 
repository on github. You can then install WHOC (again, assuming you have 
already activated your conda environment) according to:

```
git clone https://github.com/your-github-id/wind-hybrid-open-controller
pip install -e "wind-hybrid-open-controller/[develop]"
```
To contribute back to the base repository 
https://github.com/NREL/wind-hybrid-open-controller, please do the following:
- Create a branch from the base repository's `develop` branch on your fork 
containing your code changes (e.g. `your-github-id:feature/your-new-feature`)
- Open a pull request into the base repository's `NREL:develop` branch, and provide 
a description of the new/updated capabilities
- The maintainers will review your pull request and provide feedback before 
possibly merging the pull request (via the "squash and merge" method) into the
`NREL:develop` branch
- At the next release, `NREL:develop` will be merged into `NREL:main`, and your changes
contributions will appear there

For more information on what your pull request should contain, see 
[Code development](code_development.md).



(installation)=
# Installation

Hycon is _not_ designed to be used as a stand-alone package. Most likely, 
you'll want to add Hycon to an existing conda environment that contains your
simulation testbed, such as [Hercules](https://github.com/NatLabRockies/hercules). 
For example, see the 
[Hercules installation instructions](https://natlabrockies.github.io/hercules/install_instructions.html)
for how to set up an appropriate conda environment.

(installation_general_users)=
## General users

If you intend to use Hycon, but not contribute, the following lines should
be sufficient to install Hycon (presumably, after activating your conda 
environment):

```
git clone https://github.com/NatLabRockies/hycon
pip install hycon/
```

(installation_developers)=
## Developers

If you intend to contribute to Hycon, we request that your fork the Hycon 
repository on github. You can then install Hycon (again, assuming you have 
already activated your conda environment) according to:

```
git clone https://github.com/your-github-id/hycon
pip install -e "hycon/[develop]"
```
To contribute back to the base repository 
https://github.com/NatLabRockies/hycon, please do the following:
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

(installation_examples)=
## To run examples

All Hycon examples run in the [Hercules](https://github.com/NatLabRockies/hercules) simulation environment.
To run the examples, you will need to additionally install Hercules. See the 
[Hercules installation instructions](https://natlabrockies.github.io/hercules/install_instructions.html)
for details.

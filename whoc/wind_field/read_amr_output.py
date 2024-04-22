from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os

import whoc

def plot_field(plane, name):
    nlevels = len(plane.groups)
    shp = (plane.groups["level_0"].variables[name][:]).shape
    if len(shp) == 4:
        ncomp = shp[-1]
    else:
        ncomp = 1
    fig, axs = plt.subplots(nrows=nlevels, ncols=ncomp, sharex=True, figsize=(8*ncomp,6*nlevels), squeeze=False)
    fig.suptitle(f"{name}_{plane.name}", fontsize=20)
    for component in range(ncomp):
        for i, lev in enumerate(plane.groups):
            fld = plane.groups[lev].variables[name][:]
            lo = plane.groups[lev].variables["lo"][:]
            hi = plane.groups[lev].variables["hi"][:]

            if ncomp == 1:
                arr = fld[0, :, :].T
            else:
                arr = fld[0, :, :, component].T
            axs[i, component].imshow(
                arr,
                extent=[lo[0], hi[0], lo[1], hi[1]],
                origin="lower",
                aspect="auto",
            )
            axs[i, component].set_title(lev)
    plt.savefig(f"{name}_{plane.name}.png")

if __name__ == "__main__":
    # TODO what's a level, how do I get the freestream wind speed from this, did I run the precursor simulation properly?
    # TODO how can I feed amr a freestream wind speed via helics

    output_path = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "post_processing", "samplingPlane00000.nc")

    # load file and inspect top level
    rg = Dataset(output_path, "r")
    print(rg)

    # Looping through the planes
    for grp in rg.groups:
        print(f"""Accessing {grp}:""")
        print(rg.groups[grp])

    # Inspect the output times
    print(rg.variables["time"][:])

    # Looping through the AMR levels in a given plane
    plane = "ylo"
    for grp in rg.groups[plane].groups:
        print(f"""Accessing {grp} in plane {plane}:""")
        print(rg.groups[plane].groups[grp])

    # An example of plotting the data in the different planes and levels

    for plane in ["ylo", "xlo"]:
        plot_field(rg.groups[plane], "velocity")
        plot_field(rg.groups[plane], "temperature")
    
    # TODO get freestream velocity: u and v components
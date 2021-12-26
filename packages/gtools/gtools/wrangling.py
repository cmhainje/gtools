import numpy as np
import h5py


def find(f):
    """Finds the snapshot at the given path.

    If given path is a snapshot itself, returns the given argument. If the given
    path is a directory, returns the last snapshot found either in the directory
    itself or in a subdirectory called "output".

    Parameters
    ----------
    f : str
        Given snapshot file/directory.

    Returns
    -------
    str
        Actual snapshot filepath.

    Raises
    ------
    ValueError
        If no snapshot can be found.
    """

    from glob import glob
    from os.path import join, isfile, isdir

    def _404():
        raise ValueError(f"No snapshot found for input {f}.")

    if isfile(f):
        if f.lower().endswith(".hdf5"):
            return f
        else:
            _404()

    elif isdir(f):

        def _strip(x):
            dot = x.rfind(".")
            und = x[:dot].rfind("_")
            return int(x[und + 1 : dot])

        # look in given directory
        cur = glob(join(f, "snapshot_*.hdf5"))
        if len(cur) > 0:
            return sorted(cur, reverse=True, key=_strip)[0]

        # look in subdirectory "output"
        out = glob(join(f, "output", "snapshot_*.hdf5"))
        if len(out) > 0:
            return sorted(out, reverse=True, key=_strip)[0]

        _404()

    else:
        _404()


def make_hdf5(filename, n_halo=0, n_disk=0, n_bulge=0, filemode="w"):
    """Make a new HDF5 file at the given filename with header and dummy
    values set up for GIZMO.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to open.

    n_halo : float, optional
        The number of DM particles. Defaults to 0.

    n_disk : float, optional
        The number of disk particles. Defaults to 0.

    n_bulge : float, optional
        The number of bulge particles. Defaults to 0.

    filemode : str, optional
        The mode to use when opening the file. Defaults to "w". Must have
        write permissions (e.g. 'r' will fail).

    Returns
    -------
    h5py.File
        The hdf5 file that has been opened. Be sure to close it!
    """

    f = h5py.File(filename, filemode)

    # Write the header
    h = f.create_group("Header")
    npart = np.array([0, n_halo, n_disk, n_bulge, 0, 0], dtype=int)
    h.attrs["NumPart_ThisFile"] = npart
    h.attrs["NumPart_Total"] = npart
    h.attrs["NumPart_Total_HighWord"] = 0 * npart

    # Write dummy values to required header attributes
    # (will be filled in my GIZMO)
    h.attrs["MassTable"] = np.zeros(6)
    h.attrs["Time"] = 0.0
    h.attrs["Redshift"] = 0.0
    h.attrs["BoxSize"] = 1.0
    h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 1.0
    h.attrs["OmegaLambda"] = 0.0
    h.attrs["HubbleParam"] = 1.0
    h.attrs["Flag_Sfr"] = 0
    h.attrs["Flag_Cooling"] = 0
    h.attrs["Flag_StellarAge"] = 0
    h.attrs["Flag_Metals"] = 0
    h.attrs["Flag_Feedback"] = 0
    h.attrs["Flag_DoublePrecision"] = 0
    h.attrs["Flag_IC_Info"] = 0

    return f


def move(g, pos, vel):
    """Moves the given galaxy to the desired center-of-mass position and
    velocity.

    Parameters
    ----------
    g : h5py File or dict
        The h5py File object containing the galaxy data. Alternatively, can be a
        dict containing the same `PartType` datasets.
    pos : array, shape (3,)
        The desired position of the galaxy center-of-mass.
    vel : array, shape (3,)
        The desired velocity of the galaxy center-of-mass.

    Returns
    -------
    dict
        A dict containing the particle data for the moved galaxy.
    """

    part_types = [k for k in g.keys() if "PartType" in k]
    all_p = np.concatenate([g[p]["Coordinates"] for p in part_types])
    all_v = np.concatenate([g[p]["Velocities"] for p in part_types])
    all_m = np.concatenate([g[p]["Masses"] for p in part_types])

    # find the center of mass position and velocity
    com = np.average(all_p, weights=all_m, axis=0)
    cov = np.average(all_v, weights=all_m, axis=0)

    out = dict()
    for p in part_types:
        out[p] = {
            "Coordinates": g[p]["Coordinates"] + pos - com,
            "Velocities": g[p]["Velocities"] + vel - cov,
            "ParticleIDs": g[p]["ParticleIDs"],
            "Masses": g[p]["Masses"],
        }

    return out


def combine(gs):
    """Combines the given galaxies into one.

    Parameters
    ----------
    gs : iterable of galaxy-like objects
        A list of the h5py File or dict objects containing the particle data for
        the galaxies to be combined.

    Returns
    -------
    dict
        A dict containing the particle data for the merged galaxies.
    """

    out = dict()
    for g in gs:
        for p in g.keys():
            if "PartType" not in p:
                continue

            if p not in out:
                out[p] = {
                    "ParticleIDs": g[p]["ParticleIDs"],
                    "Coordinates": g[p]["Coordinates"],
                    "Velocities": g[p]["Velocities"],
                    "Masses": g[p]["Masses"],
                }

            else:
                # find max particle ID from previously-added galaxies
                k = "ParticleIDs"
                max_ID = max([np.amax(out[p][k]) for p in out.keys()])
                out[p][k] = np.concatenate([out[p][k], g[p][k] + max_ID])

                # append other galaxy data
                for k in ["Coordinates", "Velocities", "Masses"]:
                    out[p][k] = np.concatenate([out[p][k], g[p][k]])

    return out

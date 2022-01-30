# gtools

Tools for GIZMO analysis.

This repository contains two Python packages: gtools and cpg.

# Installation

Clone this repository. Then, `cd` into it.

From there,
```bash
$ cd packages/
$ pip install -e gtools
$ pip install -e cpg
```

# gtools

The core of the `gtools` suite. Contains...
- `Reader`: a class that reads the snapshots in a given directory and allows for
  faster, easier access to their data.
- `HaloFinder`: a class which executes the AMIGA Halo Finder unbinding procedure
  in order to determine whether particles are bound to a halo.
- `wrangling`: a module which provides methods for finding GIZMO snapshots,
  making new GIZMO-compatible HDf5 files, and moving/combining whole
  galaxies or snapshot files.
- `models`: a subpackage that provides convenient methods for computing virial,
  NFW, and Hernquist parameters for halo modeling.

# cpg

Connor's Paramfile Generator. It does what it says on the tin! Contains methods
for making GIZMO and GalIC paramfiles (as well as corresponding SLURM batch
scripts) from the command-line within a given directory, automatically filling
out all of the relevant paths in the files as well as providing the opportunity
to set a few key parameters (like GIZMO evolution time). (Note: the GalIC
paramfile generator is not yet live.)

To make a GIZMO paramfile and SLURM batch script in the current directory,
type...
```bash
$ python -m cpg.gizmo
```

To see all the available options, use...
```bash
$ python -m cpg.gizmo --help
```

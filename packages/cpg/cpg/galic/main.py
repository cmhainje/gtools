import argparse
import os
from . import param, slurm


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=".")
    ap.add_argument("--name", default="gen")
    ap.add_argument("--partition", default="all")
    ap.add_argument("--memory", default="100G")
    ap.add_argument("--jobtime", default="1-00:00:00")
    ap.add_argument("--param", default="./galic.param")
    ap.add_argument("--output", default="./output")
    ap.add_argument("--redshift", default=0, type=float)
    ap.add_argument("--c", default=10, type=float)
    ap.add_argument("--n_halo", type=int, required=True)
    ap.add_argument("--n_disk", type=int, required=True)
    ap.add_argument("--n_bulge", type=int, required=True)
    ap.add_argument("--M_halo", default=0, type=float, help="(M200) in M_sun")
    ap.add_argument("--M_disk", default=0, type=float, help="in M_sun")
    ap.add_argument("--M_bulge", default=0, type=float, help="in M_sun")
    ap.add_argument("--disk_scale", default=0, type=float, help="in kpc")
    ap.add_argument("--disk_height", default=0, type=float, help="in kpc")
    ap.add_argument("--bulge_scale", default=0, type=float, help="in kpc")
    args = ap.parse_args()

    args.location = os.path.abspath(args.location)
    args.param = os.path.abspath(args.param)
    return args


def main():
    args = parse()
    param.write(param.process(args))
    slurm.write(args)


if __name__ == "__main__":
    main()

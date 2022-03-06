import argparse
import os
from . import param, slurm


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=".")
    ap.add_argument("--icpath", default="./ic")
    ap.add_argument("--output", default="./output")
    ap.add_argument("--simtime", default="10.0")
    ap.add_argument("--snaptime", default="0.1")
    ap.add_argument("--dm_cross_section", default="0")
    ap.add_argument("--dm_velocity_scale", default="0")
    ap.add_argument("--name", default="sim")
    ap.add_argument("--partition", default="all")
    ap.add_argument("--memory", default="50G")
    ap.add_argument("--jobtime", default="1-00:00:00")
    ap.add_argument("--ntasks", default="25")
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--gizmo", default="/home/chainje/gizmo-public/GIZMO")
    args = ap.parse_args()

    args.location = os.path.abspath(args.location)
    args.icpath = os.path.abspath(args.icpath)
    args.output = os.path.abspath(args.output)
    args.gizmo = os.path.abspath(args.gizmo)
    args.param = os.path.join(args.location, "gizmo.param")

    return args


def main():
    args = parse()
    param.write(args)
    slurm.write(args)


if __name__ == "__main__":
    main()

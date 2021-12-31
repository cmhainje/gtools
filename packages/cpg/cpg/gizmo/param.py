import argparse
import os
from jinja2 import Environment, PackageLoader


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=".")
    ap.add_argument("--icpath", default="./ic")
    ap.add_argument("--output", default="./output")
    ap.add_argument("--simtime", default="10.0")
    ap.add_argument("--dm_cross_section", default="0")
    ap.add_argument("--dm_velocity_scale", default="0")
    args = ap.parse_args()

    args.location = os.path.abspath(args.location)
    args.icpath = os.path.abspath(args.icpath)
    args.output = os.path.abspath(args.output)
    return args


def write(args):
    file_loader = PackageLoader("cpg.gizmo", "templates")
    env = Environment(loader=file_loader)
    template = env.get_template("param.txt")
    output = template.render(
        icpath=args.icpath,
        output=args.output,
        time=args.simtime,
        dm_cross_section=args.dm_cross_section,
        dm_velocity_scale=args.dm_velocity_scale,
    )

    with open(os.path.join(args.location, "gizmo.param"), "w") as f:
        f.write(output)


def main():
    write(parse())


if __name__ == "__main__":
    main()

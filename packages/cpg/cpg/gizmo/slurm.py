import argparse
import os
from jinja2 import Environment, PackageLoader


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=".")
    ap.add_argument("--name", default="sim")
    ap.add_argument("--partition", default="all")
    ap.add_argument("--memory", default="50G")
    ap.add_argument("--jobtime", default="1-00:00:00")
    ap.add_argument("--ntasks", default="25")
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--gizmo", default="/home/chainje/gizmo-public/GIZMO")
    ap.add_argument("--param", default="./gizmo.param")
    args = ap.parse_args()

    args.location = os.path.abspath(args.location)
    args.gizmo = os.path.abspath(args.gizmo)
    args.param = os.path.abspath(args.param)
    return args


def write(args):
    file_loader = PackageLoader("cpg.gizmo", "templates")
    env = Environment(loader=file_loader)
    template = env.get_template("slurm.txt")
    output = template.render(
        job_n=args.name,
        job_p=args.partition,
        job_m=args.memory,
        job_t=args.jobtime,
        ntasks=args.ntasks,
        exclusive=args.exclusive,
        gizmopath=args.gizmo,
        parampath=args.param,
    )

    with open(os.path.join(args.location, "run_gizmo.sh"), "w") as f:
        f.write(output)


def main():
    write(parse())


if __name__ == "__main__":
    main()

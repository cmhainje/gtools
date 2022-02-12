import argparse
import os
from jinja2 import Environment, PackageLoader

import astropy.units as u
from astropy.cosmology import Planck13
from astropy.constants import G
from gtools.models.virial import Virial
from gtools.models.halos import NFW


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=".")
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

    return process(args)


def process(args):
    args.location = os.path.abspath(args.location)
    args.output = os.path.abspath(args.output)

    M_tot = args.M_halo + args.M_disk + args.M_bulge
    vir = Virial(cosmology=Planck13, redshift=args.redshift, overdensity=200)
    args.hubble = vir.hubble.to(1 / u.s).value
    args.v200 = vir.compute_V_from_M(M_tot * u.M_sun).value

    if args.n_disk == 0 or args.M_disk == 0:
        args.n_disk = 0
        args.M_disk = 0
        args.Mfrac_disk = 0

        args.disk_spinfrac = 0
        args.spin_param = 0
        args.disk_height = 0

    else:
        args.Mfrac_disk = args.M_disk / (args.M_halo + args.M_disk + args.M_bulge)
        if args.disk_scale == 0:
            raise ValueError("disk scale length unspecified")
        if args.disk_height == 0:
            raise ValueError("disk height unspecified")

        # from disk_scale and disk_height, compute lambda and DiskHeight
        args.disk_spinfrac = args.Mfrac_disk  # spin fraction
        r200 = vir.compute_R_from_M(M_tot * u.M_sun).value
        args.spin_param = args.disk_scale * 2 ** (0.5) / r200
        args.disk_height = args.disk_height / args.disk_scale

    if args.n_bulge == 0 or args.M_bulge == 0:
        args.n_bulge = 0
        args.M_bulge = 0
        args.Mfrac_bulge = 0
        args.bulge_scale = 0
    else:
        args.Mfrac_bulge = args.M_bulge / (args.M_halo + args.M_disk + args.M_bulge)
        if args.bulge_scale == 0:
            raise ValueError("bulge scale unspecified")
        # convert bulge scale length from kpc to units of halo scale length
        nfw_scale = NFW.from_M200_c200(
            args.M_halo, args.c, cosmology=Planck13, redshift=args.redshift
        ).r_s
        args.bulge_scale = args.bulge_scale / nfw_scale

    return args


def write(args):
    file_loader = PackageLoader("cpg.galic", "templates")
    env = Environment(loader=file_loader)
    template = env.get_template("param.txt")
    output = template.render(
        output=args.output,
        hubble=args.hubble,
        c200=args.c,
        v200=args.v200,
        spin_param=args.spin_param,
        massfrac_disk=args.Mfrac_disk,
        massfrac_bulge=args.Mfrac_bulge,
        spinfrac_disk=args.disk_spinfrac,
        disk_height=args.disk_height,
        bulge_scale=args.bulge_scale,
        n_halo=args.n_halo,
        n_disk=args.n_disk,
        n_bulge=args.n_bulge,
    )

    with open(os.path.join(args.location, "galic.param"), "w") as f:
        f.write(output)


def main():
    write(parse())


if __name__ == "__main__":
    main()

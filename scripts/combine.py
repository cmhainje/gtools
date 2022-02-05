import h5py
import numpy as np
import argparse
import gtools.wrangling as w


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "inputs", type=str, nargs="+", help="Input files (as many as you want)."
    )
    ap.add_argument(
        "-o", "--output", type=str, default="./ic.hdf5", help="Output filename."
    )

    ap.add_argument(
        "--pos",
        type=float,
        nargs="*",
        help=(
            "Desired center-of-mass positions. "
            "Parsed as x1 y1 z1 x2 y2 z2 ... "
            "Missing values are filled with 0."
        ),
    )
    ap.add_argument(
        "--vel",
        type=float,
        nargs="*",
        help=(
            "Desired center-of-mass velocities. "
            "Parsed as x1 y1 z1 x2 y2 z2 ... "
            "Missing values are filled with 0."
        ),
    )
    ap.add_argument(
        "--r_trunc",
        type=float,
        nargs="*",
        help=(
            "Desired truncation radius. "
            "Parsed as rt1 rt2 ... "
            "Missing values are filled with 0 (no truncation)."
        ),
    )

    return ap.parse_args()


def main():
    args = parse_args()

    fs = [w.find(i) for i in args.inputs]
    gals = [h5py.File(f, "r") for f in fs]
    print(f"Snapshots found and read...")
    for f in fs:
        print(f" -> {f}")

    # move each galaxy
    des_pos = args.pos
    for _ in range(3 * len(gals) - len(des_pos)):
        des_pos.append(0)

    des_vel = args.vel
    for _ in range(3 * len(gals) - len(des_vel)):
        des_vel.append(0)

    des_rtrunc = args.r_trunc
    for _ in range(len(gals) - len(des_rtrunc)):
        des_rtrunc.append(0)

    moveds = []
    for i, gal in enumerate(gals):
        pos = np.array(des_pos[i * 3 : (i + 1) * 3])
        vel = np.array(des_vel[i * 3 : (i + 1) * 3])
        moved = w.move(gal, pos, vel)
        print(f"Galaxy {i+1} moved to...")
        print(f" ..position {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}.")
        print(f" ..velocity {vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}.")

        if des_rtrunc[i] != 0:
            moved = w.truncate(gal, des_rtrunc[i])
            print(f"Galaxy {i+1} truncated at a radius of {des_rtrunc[i]:.3f} kpc.")

        moveds.append(moved)

    # make output file
    n_parts = dict(n_halo=0, n_disk=0, n_bulge=0)
    for i, n in enumerate(n_parts.keys()):
        p = f"PartType{i+1}"
        for gal in moveds:
            n_parts[n] += len(gal[p]["ParticleIDs"]) if p in gal else 0
    out = w.make_hdf5(args.output, **n_parts)

    print(f"New, empty galaxy file made at {args.output}.")
    print(f"  Number of halo particles:  {n_parts['n_halo']}")
    print(f"  Number of disk particles:  {n_parts['n_disk']}")
    print(f"  Number of bulge particles: {n_parts['n_bulge']}")

    # combine galaxies and save to output file
    com = w.combine(moveds)
    for p in com.keys():
        for k in com[p].keys():
            out.create_dataset(f"{p}/{k}", data=com[p][k])
    print("Data loaded into output file.")

    for gal in gals:
        gal.close()
    out.close()
    print("Done!")


if __name__ == "__main__":
    main()

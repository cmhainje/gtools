import h5py
import numpy as np
import argparse
import gtools.wrangling as w


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Input filename.")
    ap.add_argument(
        "-o", "--output", type=str, default="./ic.hdf5", help="Output filename."
    )
    ap.add_argument(
        "--pos",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Desired center-of-mass position.",
    )
    ap.add_argument(
        "--vel",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Desired center-of-mass velocity.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    f = w.find(args.input)
    gal = h5py.File(f, "r")
    print(f"Snapshot found and read at {f}.")

    # make output file
    n_halo = len(gal["PartType1"]["ParticleIDs"]) if "PartType1" in gal else 0
    n_disk = len(gal["PartType2"]["ParticleIDs"]) if "PartType2" in gal else 0
    n_bulge = len(gal["PartType3"]["ParticleIDs"]) if "PartType3" in gal else 0
    out = w.make_hdf5(args.output, n_halo=n_halo, n_disk=n_disk, n_bulge=n_bulge)
    print(f"New galaxy file made at {args.output}.")
    print(f"  Number of halo particles:  {n_halo}")
    print(f"  Number of disk particles:  {n_disk}")
    print(f"  Number of bulge particles: {n_bulge}")

    # fill output file with moved galaxy particle data
    moved = w.move(gal, np.array(args.pos), np.array(args.vel))
    print(f"Moved to...")
    print(f" ..position {args.pos[0]:.1f}, {args.pos[1]:.1f}, {args.pos[2]:.1f}.")
    print(f" ..velocity {args.vel[0]:.1f}, {args.vel[1]:.1f}, {args.vel[2]:.1f}.")

    for p in moved.keys():
        for k in moved[p].keys():
            out.create_dataset(f"{p}/{k}", data=moved[p][k])
    print("Data loaded into output file.")

    gal.close()
    out.close()
    print("Done!")


if __name__ == "__main__":
    main()

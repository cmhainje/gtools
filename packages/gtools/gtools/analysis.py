import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from .halofinder import HaloFinder


def compute_bounds(
    g, part_types=["sgr_dark"], verbose=True, strip_permanent=True, r_trunc=None
):
    coms, covs, nums, rads, bounds = [], [], [], [], []
    bound = np.concatenate(
        [np.ones(len(g.get_mass(0, p)), dtype=bool) for p in part_types]
    )

    if r_trunc is not None:
        pos = np.concatenate([g.get_pos(0, p) for p in part_types])
        truncate = np.linalg.norm(pos, axis=1) > r_trunc
        bound[truncate] = False

    if verbose:
        print("index\tGyr\tbound particles")

    for i in range(len(g.snapshots)):
        pos = np.concatenate([g.get_pos(i, p) for p in part_types])
        vel = np.concatenate([g.get_vel(i, p) for p in part_types])
        mass = np.concatenate([g.get_mass(i, p) for p in part_types])

        hf = HaloFinder(
            mass * u.M_sun,
            pos * u.kpc,
            vel * u.km / u.s,
            init_bound=bound if strip_permanent else None,
        )
        hf.unbind()
        bound = hf.unsort_bound()
        bounds.append(bound)

        coms.append(np.average(pos[bound], axis=0, weights=mass[bound]))
        covs.append(np.average(vel[bound], axis=0, weights=mass[bound]))
        nums.append(np.count_nonzero(bound))
        rads.append(np.amax(hf.r[hf.bound]).value)

        if verbose:
            print(f"{i:2d}\t{g.get_time(i):.3f}\t{np.count_nonzero(bound)}")

    coms = np.array(coms)
    covs = np.array(covs)
    nums = np.array(nums)
    rads = np.array(rads)
    return dict(bounds=bounds, coms=coms, covs=covs, nums=nums, rads=rads)


def scatter_plot(g, snap, bound, part_types, x_dim=0, y_dim=2, lim=500):

    pos = np.concatenate([g.get_pos(snap, p) for p in part_types])
    p_indices = np.zeros(len(pos), dtype=int)
    last_idx = 0
    for p in part_types:
        last_idx += len(g.get_mass(snap, p))
        p_indices[last_idx:] += 1

    fig, axs = plt.subplots(1, len(part_types), figsize=(len(part_types) * 5, 5))

    for i, p in enumerate(part_types):
        mask = p_indices == i
        axs[i].plot(
            pos[~bound & mask, x_dim], pos[~bound & mask, y_dim], ".", markersize=3
        )
        axs[i].plot(
            pos[bound & mask, x_dim], pos[bound & mask, y_dim], ".", markersize=3
        )
        axs[i].set_title(p)

    for ax in axs.flat:
        ax.set_xlim(-lim, +lim)
        ax.set_ylim(-lim, +lim)

    fig.suptitle(f"{g.get_time(snap):.3f} Gyr")
    fig.tight_layout()
    return fig


def subplot_trajectory(axs, g, data, part_types=["sgr_dark"], c=None, ls="-"):
    times = [g.get_time(i) for i in range(len(g.snapshots))]

    coms = data["coms"]
    covs = data["covs"]
    nums = data["nums"]
    rads = data["rads"]
    bounds = data["bounds"]

    pos = np.concatenate([g.get_pos(0, p) for p in part_types])
    p_indices = np.zeros(len(pos), dtype=int)
    last_idx = 0
    for p in part_types:
        last_idx += len(g.get_mass(0, p))
        p_indices[last_idx:] += 1

    nums = []
    for snap in range(len(times)):
        bound = bounds[snap]
        nums.append(
            [np.count_nonzero(bound[p_indices == i]) for i in range(len(part_types))]
        )
    nums = np.array(nums)
    masses = np.array([g.get_mass(0, p)[0] for p in part_types])
    m_tot = (nums * masses).sum(1)

    axs[0, 0].plot(times, m_tot / m_tot[0], c=c, ls="solid", lw=1.2)
    for i, p in enumerate(part_types):
        linestyle = ["dashed", "dotted", "dashdot"][i % 3]
        axs[0, 0].plot(times, nums[:, i] / nums[0, i], c=c, ls=linestyle, lw=0.8)
    axs[0, 0].set_xlabel("Time [Gyr]")
    axs[0, 0].set_ylabel("$M(0) / M(t)$")
    axs[0, 0].set_title("Bound mass")

    axs[1, 0].plot(times, rads, c=c, ls=ls)
    axs[1, 0].set_xlabel("Time [Gyr]")
    axs[1, 0].set_ylabel("$R_{trunc}$ [kpc]")
    axs[1, 0].set_title("Max radius of bound particles")
    axs[1, 0].set_ylim(0, 500)

    axs[0, 1].plot(coms[:, 0], coms[:, 2], c=c, ls=ls)
    axs[0, 1].set_xlabel("$x$ [kpc]")
    axs[0, 1].set_ylabel("$z$ [kpc]")
    axs[0, 1].set_title("Position")

    axs[1, 1].plot(covs[:, 0], covs[:, 2], c=c, ls=ls)
    axs[1, 1].set_xlabel("$v_x$ [km/s]")
    axs[1, 1].set_ylabel("$v_z$ [km/s]")
    axs[1, 1].set_title("Velocity")

    axs[0, 2].plot(times, np.sqrt((coms ** 2).sum(1)), c=c, ls=ls)
    axs[0, 2].set_xlabel("Time [Gyr]")
    axs[0, 2].set_ylabel("Distance [kpc]")
    axs[0, 2].set_title("Distance from center")

    axs[1, 2].plot(times, np.sqrt((covs ** 2).sum(1)), c=c, ls=ls)
    axs[1, 2].set_xlabel("Time [Gyr]")
    axs[1, 2].set_ylabel("Speed [km/s]")
    axs[1, 2].set_title("Magnitude of velocity")


def plot_trajectory(g, data, part_types=["sgr_dark"], c=None, ls="-"):
    fig, axs = plt.subplots(2, 3, figsize=(3.5 * 5, 2 * 5))
    subplot_trajectory(axs, g, data, part_types=part_types, c=c, ls=ls)
    fig.tight_layout()
    return fig

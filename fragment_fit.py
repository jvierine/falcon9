import numpy as n
import numpy as np
import matplotlib.pyplot as plt
import jcoord
from pymsis import msis
import fit_ballistic3 as fb3

import plot_fragments as pf

def merge_parent_measurements(merge_ids,fragment_times,fragment_pos,fragment_pos_err,fragment_ids):
    if isinstance(merge_ids, str):
        merge_ids = [merge_ids]
    else:
        merge_ids = list(merge_ids)

    if len(merge_ids) == 0:
        raise ValueError("merge_ids must contain at least one fragment id")

    fragment_index = {}
    for i in range(len(fragment_ids)):
        fragment_index[fragment_ids[i]] = i

    ordered_ids = []
    seen = set()
    for fid in merge_ids:
        if fid not in fragment_index:
            raise ValueError("unknown fragment id %s" % (fid))
        if fid in seen:
            continue
        ordered_ids.append(fid)
        seen.add(fid)

    merged_times = None
    merged_pos = None
    merged_pos_err = None
    used_ids = []

    for fid in ordered_ids:
        idx = fragment_index[fid]
        times = np.asarray(fragment_times[idx], dtype=float)
        pos = np.asarray(fragment_pos[idx], dtype=float)
        pos_err = np.asarray(fragment_pos_err[idx], dtype=float)

        order = np.argsort(times, kind="mergesort")
        times = times[order]
        pos = pos[order, :]
        pos_err = pos_err[order]

        if len(times) == 0:
            continue

        if merged_times is None:
            merged_times = times.copy()
            merged_pos = pos.copy()
            merged_pos_err = pos_err.copy()
            used_ids.append(fid)
            continue

        # Work backward in time: only use parent measurements that occur
        # strictly before the first already-merged child measurement.
        use = times < merged_times[0]
        if not np.any(use):
            continue

        merged_times = np.concatenate((times[use], merged_times))
        merged_pos = np.vstack((pos[use, :], merged_pos))
        merged_pos_err = np.concatenate((pos_err[use], merged_pos_err))
        used_ids.append(fid)

    if merged_times is None or len(merged_times) == 0:
        raise RuntimeError("no measurements found for merge_ids=%s" % (merge_ids))

    print("manual merge: %s" % (" -> ".join(ordered_ids)))
    print(
        "merged %s: n=%d t0=%1.2f t1=%1.2f"
        % (
            ",".join(used_ids),
            len(merged_times),
            merged_times[0],
            merged_times[-1],
        )
    )
    return(merged_times, merged_pos, merged_pos_err, used_ids)


def plot_all_fragments_with_ids(fragment_geo_pos, fragment_ids, fragment_times):
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(12,5.5),sharey=True)

    t0 = np.min([np.min(t) for t in fragment_times if len(t) > 0])

    for i, fid in enumerate(fragment_ids):
        geo = np.asarray(fragment_geo_pos[i])
        times = np.asarray(fragment_times[i], dtype=float)
        if len(geo) == 0:
            continue

        color = "C%d" % (i % 10)
        lon = geo[:,1]
        alt_km = geo[:,2]/1e3
        time_rel = times - t0

        ax0.plot(lon, alt_km, ".", color=color, alpha=0.8, markersize=4)
        ax1.plot(time_rel, alt_km, ".", color=color, alpha=0.8, markersize=4)

        ax0.text(lon[0], alt_km[0], fid, color=color, fontsize=8)
        ax1.text(time_rel[0], alt_km[0], fid, color=color, fontsize=8)

    ax0.set_xlabel("Longitude (deg)")
    ax0.set_ylabel("Altitude (km)")
    ax0.set_title("Fragments")
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax1.set_xlabel("Time since first detection (s)")
    ax1.set_title("Fragments")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig("fragment_id_overview.pdf", bbox_inches="tight")
    return(fig)


def plot_merged_measurements_lon_alt(child_id, fragment_geo_pos, merged_times, merged_pos, merged_chain):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for geo in fragment_geo_pos:
        ax.plot(
            geo[:, 1],
            geo[:, 2] / 1e3,
            ".",
            color="0.75",
            markersize=3,
            alpha=0.7,
            zorder=1,
        )

    merged_lon = []
    merged_alt_km = []
    for pos in merged_pos:
        llh = jcoord.ecef2geodetic(pos[0], pos[1], pos[2])
        merged_lon.append(llh[1])
        merged_alt_km.append(llh[2] / 1e3)

    sc = ax.scatter(
        merged_lon,
        merged_alt_km,
        c=merged_times,
        cmap="turbo",
        s=18,
        zorder=3,
        label="Merged measurements",
    )
    ax.plot(
        merged_lon,
        merged_alt_km,
        "-",
        color="tab:blue",
        linewidth=1.0,
        alpha=0.7,
        zorder=2,
    )

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Unix time (s)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("%s merged path: %s" % (child_id, " -> ".join(merged_chain)))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("merged_%s_lon_alt.pdf" % (child_id), bbox_inches="tight")
    return(fig)


# the idea is to setup what ids to fit here
# $F_1$ are $F_3$, $F_5$, $F_9$, $F_n$, $F_o$, $F_p$, $F_r$, $F_s$, $F_w$, $F_v$, $F_x$, and $F_z$.
# The child fragments of $F_2$ are $F_4$, $F_7$, $F_8$, $F_a$, $F_c$, $F_h$, $F_g$, $F_i$, $F_m$, $F_j$, $F_{\ell}$, $F_k$, $F_t$, $F_u$, $F_e$, and $F_d$. The
#f1_children = ["1","3","5","9","n","o","p","r","s","w","v","x","z"]
f1_children = ["1"]#,"3","5","9","n","o","p","r","s","w","v","x","z"]
f2_children = ["2","4","7","8","a","c","h","g","i","m","j","l","k","t","u","e","d"]

hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()
#plot_all_fragments_with_ids(fragment_geo_pos, fragment_ids, fragment_times)
#plt.show()

# Edit this list after inspecting fragment_id_overview.pdf or the overview plot.
#merge_ids=["5","1"]

merge_ids_all=[
    ["1"],
    ["2"],
    ["5","1"],
    ["9","o","1"],
    ["z","w","p","1"],
    ["t","i","7","2"],
    ["g","4","2"],
    ["k","a","7","2"]]

for merge_ids in merge_ids_all:
    merged_times, merged_pos, merged_pos_err, merged_chain = merge_parent_measurements(merge_ids,fragment_times,fragment_pos,fragment_pos_err,fragment_ids)
    plot_merged_measurements_lon_alt(",".join(merge_ids), fragment_geo_pos, merged_times, merged_pos, merged_chain)
    plt.show()

    result = fb3.fit_shared_ballistic_coefficient(
        fragment_pos=merged_pos,
        fragment_pos_err=merged_pos_err,
        fragment_times=merged_times,
        fit_ids=merged_chain,
        B0_guess=[-3,-3],
        verbose=2
    )

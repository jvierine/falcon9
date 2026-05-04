#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import plot_deco
import plot_rdmf


START_TIME = "2025-02-19T03:45:47"
END_TIME = "2025-02-19T03:46:26"
RCS_YMIN_KM = 215
RCS_YMAX_KM = 330
DOPPLER_YMIN_KM = RCS_YMIN_KM
DOPPLER_YMAX_KM = RCS_YMAX_KM
OUTPUT = Path("rcs_doppler_triptych.pdf")
DOPPLER_HDF5_FILE = Path("simone/koki/figures_miso/summary_data_kborn_hagenow_20250219_miso_ref_tx_corr.h5")
RCS_FIT_PATH = Path("ballistic_fit_sharedstart_1.h5")


def triptych_rcparams():
    return {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).parent / path


def read_doppler_from_hdf5(hdf5_path: Path = DOPPLER_HDF5_FILE):
    """Read doppler shift measurements from the summary HDF5 file."""
    hdf5_path = resolve_repo_path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        doppler = np.asarray(f["summary_doppler"][()], dtype=float)
        taxis = np.asarray(f["taxis"][()], dtype=float)
        ranges = np.asarray(f["ranges"][()], dtype=float)

    return doppler, taxis, ranges


def compute_fit_range_rcs_grid(tx="kborn", rx="hagenow", fit_path=RCS_FIT_PATH):
    """
    Convert decoded SNR to RCS using fitted trajectory geometry for each range gate.

    Each propagation-range gate is assigned the nearest time along the ballistic
    fit where the fitted bistatic propagation range intersects that gate. The
    transmitter-target and target-receiver ranges from that fitted point are
    then used for all samples in the corresponding range gate.
    """
    fit_path = resolve_repo_path(fit_path)
    decoded = plot_deco.load_decoded_power(tx=tx, rx=rx)
    snr = plot_deco.compute_snr_from_power(decoded["power"])
    range_km = np.asarray(decoded["range_km"], dtype=float)

    track_times_unix = plot_deco.load_fragment_fit_model_trajectory(fit_path=fit_path)["times_unix"]
    track_times_dt64 = np.asarray(track_times_unix * 1e9, dtype="datetime64[ns]")
    fit_geometry = plot_deco.get_fragment_fit_bragg_geometry(
        tx=tx,
        rx=rx,
        time_dt64=track_times_dt64,
        fit_path=fit_path,
    )
    fit_range_km = np.asarray(fit_geometry["propagation_range_km"], dtype=float)
    fit_r_tx_km = np.asarray(fit_geometry["r_tx_km"], dtype=float)
    fit_r_rx_km = np.asarray(fit_geometry["r_rx_km"], dtype=float)

    finite_fit = (
        np.isfinite(fit_range_km)
        & np.isfinite(fit_r_tx_km)
        & np.isfinite(fit_r_rx_km)
    )
    if np.count_nonzero(finite_fit) == 0:
        raise ValueError(f"No finite fitted trajectory geometry in {fit_path}")

    fit_range_km = fit_range_km[finite_fit]
    fit_r_tx_km = fit_r_tx_km[finite_fit]
    fit_r_rx_km = fit_r_rx_km[finite_fit]
    fit_times_unix = np.asarray(track_times_unix, dtype=float)[finite_fit]

    nearest_fit_idx = np.array(
        [int(np.nanargmin(np.abs(fit_range_km - gate_km))) for gate_km in range_km],
        dtype=int,
    )
    r_tx = fit_r_tx_km[nearest_fit_idx][:, None] * 1e3
    r_rx = fit_r_rx_km[nearest_fit_idx][:, None] * 1e3

    rcs = plot_deco.sn_plus_n_over_n_to_rcs(
        np.array(snr, dtype=float, copy=True),
        r_tx,
        r_rx,
    )
    rcs = np.where(snr < 1.2, 1e-9, rcs)

    decoded["snr"] = snr
    decoded["sn_plus_n_over_n_db"] = 10.0 * np.log10(np.maximum(snr, 1e-12))
    decoded["rcs_m2"] = rcs
    decoded["rcs_dbsm"] = 10.0 * np.log10(np.maximum(rcs, 1e-12))
    decoded["fit_range_gate_time_unix"] = fit_times_unix[nearest_fit_idx]
    decoded["fit_range_gate_r_tx_km"] = fit_r_tx_km[nearest_fit_idx]
    decoded["fit_range_gate_r_rx_km"] = fit_r_rx_km[nearest_fit_idx]
    decoded["fit_range_gate_model_range_km"] = fit_range_km[nearest_fit_idx]
    return decoded


def plot_doppler_range_panel(ax):
    """Plot doppler-range from HDF5 summary data."""
    try:
        doppler, taxis, ranges = read_doppler_from_hdf5()
        tvec = np.asarray(taxis * 1e9, dtype="datetime64[ns]")
        d_plot = -np.asarray(doppler, dtype=float)

        cmap = plt.cm.seismic.copy()
        cmap.set_bad("0.65")

        mesh = ax.pcolormesh(
            tvec,
            ranges,
            d_plot,
            cmap=cmap,
            vmin=-100,
            vmax=100,
            shading="auto",
            rasterized=True,
        )
    except Exception as e:
        print(f"Error reading from HDF5: {e}. Falling back to plot_rdmf.read_kb()")
        P, N, D, tvec, rvec = plot_rdmf.read_kb()
        sn = P / N
        d_plot = -np.asarray(D, dtype=float)
        d_plot[sn < 4.0] = np.nan

        cmap = plt.cm.seismic.copy()
        cmap.set_bad("0.65")

        mesh = ax.pcolormesh(
            tvec,
            rvec,
            d_plot.T,
            cmap=cmap,
            vmin=-100,
            vmax=100,
            shading="auto",
            rasterized=True,
        )
    
    ax.set_ylabel("Range (km)")
    ax.set_ylim(DOPPLER_YMIN_KM, DOPPLER_YMAX_KM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.tick_params(top=False, right=True)
    return mesh


def plot_optical_doppler_panel(ax):
    """Plot optical fragment doppler observations."""
    fragment_aspects, fragment_dops, fragment_range, fragment_dt = plot_deco.get_fragment_info(
        tx="kborn",
        rx="hagenow",
    )
    hgt_count, hgt_count_all, fragment_ids, *_ = plot_rdmf.pf.get_fragments()

    colors = {"1": "#cb181d", "2": "#2171b5"}
    for i, fragment_id in enumerate(fragment_ids):
        if fragment_id not in {"1", "2"}:
            continue
        ax.plot(
            fragment_dt[i],
            fragment_dops[i]*-1,
            ".",
            color=colors[fragment_id],
            markersize=3.0,
            label=rf"$F_{fragment_id}$",
        )

    ax.set_ylabel("Doppler (Hz)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.tick_params(top=False, right=True)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")



def main():
    parser = argparse.ArgumentParser(
        description="Make a 3-panel publication figure with RCS and Doppler panels."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help=f"Output PDF path (default: {OUTPUT})",
    )
    parser.add_argument(
        "--fit-path",
        type=Path,
        default=RCS_FIT_PATH,
        help=f"Ballistic fit HDF5 used for RCS range geometry (default: {RCS_FIT_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the figure interactively after saving.",
    )
    args = parser.parse_args()
    fit_path = resolve_repo_path(args.fit_path)

    start_dt = datetime.fromisoformat(START_TIME).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(END_TIME).replace(tzinfo=timezone.utc)
    shared_xlim = mdates.date2num([start_dt, end_dt])

    with plt.rc_context(triptych_rcparams()):
        fig = plt.figure(figsize=(5.2, 6.0), constrained_layout=True)
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.0, 0.05],
            height_ratios=[1.0, 1.0, 0.9],
        )
        ax0 = fig.add_subplot(gs[0, 0])
        cax0 = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
        cax1 = fig.add_subplot(gs[1, 1])
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        cax2 = fig.add_subplot(gs[2, 1])
        cax2.axis("off")
        axes = [ax0, ax1, ax2]

        rcs_grid = compute_fit_range_rcs_grid(
            tx="kborn",
            rx="hagenow",
            fit_path=fit_path,
        )
        rcs_result = plot_deco.plot_decoded(
            tx="kborn",
            rx="hagenow",
            start_time=START_TIME,
            end_time=END_TIME,
            ymin=RCS_YMIN_KM,
            ymax=RCS_YMAX_KM,
            ax=axes[0],
            add_colorbar=False,
            show=False,
            output_filename=None,
            title=None,
            precomputed_grid=rcs_grid,
            fit_path=fit_path,
        )
        cb0 = fig.colorbar(rcs_result["mesh"], cax=cax0)
        cb0.set_label("RCS (dBsm)")

        mesh1 = plot_doppler_range_panel(axes[1])
        cb1 = fig.colorbar(mesh1, cax=cax1)
        cb1.set_label("Doppler shift (Hz)")

        plot_optical_doppler_panel(axes[2])

        # --- Save panel data into a sidecar HDF5 in simone/ ---
        try:
            simone_dir = Path(__file__).parent / "simone"
            simone_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = simone_dir / "rcs_doppler_triptych_sidecar.h5"
            summary_h5 = resolve_repo_path(DOPPLER_HDF5_FILE)

            # gather rcs panel data
            rcs_times = rcs_result.get("times_datetime64")
            rcs_range_km = rcs_result.get("range_km")
            rcs_dbsm = rcs_result.get("rcs_dbsm")

            # gather optical fragment data
            frag_aspects, frag_dops, frag_range, frag_dts = plot_deco.get_fragment_info(tx="kborn", rx="hagenow")
            _, _, fragment_ids, *_ = plot_rdmf.pf.get_fragments()

            with h5py.File(sidecar_path, "w") as fh:
                fh.attrs["start_time"] = START_TIME
                fh.attrs["end_time"] = END_TIME
                fh.attrs["fit_path"] = str(fit_path)

                grp_rcs = fh.create_group("rcs")
                if rcs_times is not None:
                    grp_rcs.create_dataset("times_ns", data=rcs_times.astype("datetime64[ns]").astype("int64"))
                grp_rcs.create_dataset("range_km", data=rcs_range_km)
                grp_rcs.create_dataset("rcs_dbsm", data=rcs_dbsm)

                # doppler summary
                if summary_h5.exists():
                    with h5py.File(summary_h5, "r") as sf:
                        ranges = np.asarray(sf["ranges"][()], dtype=float)
                        taxis = np.asarray(sf["taxis"][()], dtype=float)
                        summary_dop = np.asarray(sf["summary_doppler"][()], dtype=float)

                    grp_dop = fh.create_group("doppler_summary")
                    grp_dop.create_dataset("ranges_m", data=ranges)
                    grp_dop.create_dataset("taxis_s", data=taxis)
                    # save the same sign convention used for plotting (we negated in plotting)
                    grp_dop.create_dataset("summary_doppler_hz", data=-1.0 * summary_dop)

                # optical fragments
                grp_opt = fh.create_group("optical_fragments")
                for idx, fid in enumerate(fragment_ids):
                    sub = grp_opt.create_group(f"F_{fid}")
                    if idx < len(frag_dops):
                        sub.create_dataset("doppler_hz", data=np.asarray(frag_dops[idx], dtype=float))
                    if idx < len(frag_dts):
                        times_ns = np.asarray(frag_dts[idx], dtype="datetime64[ns]").astype("int64")
                        sub.create_dataset("times_ns", data=times_ns)
                    if idx < len(frag_range):
                        sub.create_dataset("range_km", data=np.asarray(frag_range[idx], dtype=float))
        except Exception as e:
            print(f"Warning: failed to write sidecar HDF5: {e}")

        axes[0].set_xlabel("")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Range (km)")
        axes[1].tick_params(labelleft=True)
        axes[2].set_xlabel("Time (UTC)")

        for ax in axes[:2]:
            ax.set_xlim(shared_xlim)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
            ax.tick_params(axis="x", rotation=30, labelbottom=False)
        axes[2].set_xlim(shared_xlim)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
        axes[2].tick_params(axis="x", rotation=30)

        panel_labels = ["a)", "b)", "c)"]
        for ax, label in zip(axes, panel_labels):
            ax.text(
                0.02,
                0.98,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
            )

        fig.savefig(args.output, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {args.output}")
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()

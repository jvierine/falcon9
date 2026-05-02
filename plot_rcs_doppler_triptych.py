#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse

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


def plot_doppler_range_panel(ax):
    P, N, D, tvec, rvec = plot_rdmf.read_kb()
    sn = P / N
    d_plot = D.copy()
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
        "--no-show",
        action="store_true",
        help="Do not open the figure interactively after saving.",
    )
    args = parser.parse_args()

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
        )
        cb0 = fig.colorbar(rcs_result["mesh"], cax=cax0)
        cb0.set_label("RCS (dBsm)")

        mesh1 = plot_doppler_range_panel(axes[1])
        cb1 = fig.colorbar(mesh1, cax=cax1)
        cb1.set_label("Doppler shift (Hz)")

        plot_optical_doppler_panel(axes[2])

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

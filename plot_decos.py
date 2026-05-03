#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import plot_deco


DEFAULT_PANEL_RANGE_LIMITS_KM = [
    (240.0, 400.0),  # a
    (440.0, 560.0),  # b
    (250.0, 500.0),  # c
    (320.0, 480.0),  # d
    (420.0, 510.0),  # e
    (200.0, 360.0),  # f
    (260.0, 360.0),  # g
    (210.0, 360.0),  # h
]

SUMMARY_DIR = Path(__file__).with_name("simone").joinpath("decoded_summaries")
SUMMARY_FILENAME_TEMPLATE = "snr_grid_{tx}_{rx}.npz"


def get_summary_cache_path(tx, rx):
    return SUMMARY_DIR / SUMMARY_FILENAME_TEMPLATE.format(tx=tx, rx=rx)


def load_summary_cache(tx, rx):
    cache_path = get_summary_cache_path(tx, rx)
    if not cache_path.exists():
        return None

    with np.load(cache_path) as data:
        times_unix = np.asarray(data["times_unix"], dtype=float)
        times_datetime64 = np.array(times_unix * 1e9, dtype="datetime64[ns]")
        return {
            "times_unix": times_unix,
            "times_datetime64": times_datetime64,
            "range_km": np.asarray(data["range_km"], dtype=float),
            "sn_plus_n_over_n_db": np.asarray(data["sn_plus_n_over_n_db"], dtype=float),
        }


def load_plot_grid(tx, rx, use_summary_cache=False, time_smooth_samples=None, time_smooth_kernel=None):
    """Load decoded grid for plotting.

    If use_summary_cache is True and a summary exists, return it. Otherwise
    compute the full decoded grid. Optional smoothing parameters are forwarded
    to compute_rcs_grid when computing the full grid. For summary-cache data
    smoothing (if requested) is applied later in `plot_all_links` where the
    cached dB grid is available.
    """
    if use_summary_cache:
        summary = load_summary_cache(tx, rx)
        if summary is not None:
            return summary
    return plot_deco.compute_rcs_grid(
        tx=tx,
        rx=rx,
        time_smooth_samples=time_smooth_samples,
        time_smooth_kernel=time_smooth_kernel,
    )


def parse_panel_ranges(value):
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(entries) != 8:
        raise argparse.ArgumentTypeError("Expected eight comma-separated range pairs for panels a-h.")

    ranges = []
    for entry in entries:
        parts = entry.replace(":", "-").split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid panel range '{entry}'. Use min-max.")
        ymin = float(parts[0])
        ymax = float(parts[1])
        if ymax <= ymin:
            raise argparse.ArgumentTypeError(f"Invalid panel range '{entry}': ymax must exceed ymin.")
        ranges.append((ymin, ymax))
    return ranges


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a full-page 2x4 publication-quality S/N+1 figure for all eight radar links.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="UTC start time, e.g. 2025-02-19T03:44:00",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="UTC end time, e.g. 2025-02-19T03:48:00",
    )
    parser.add_argument(
        "--min-range-km",
        type=float,
        default=100.0,
        help="Minimum propagation range in km.",
    )
    parser.add_argument(
        "--max-range-km",
        type=float,
        default=600.0,
        help="Maximum propagation range in km.",
    )
    parser.add_argument(
        "--output",
        default="snr_all_links_fullpage.pdf",
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--panel-ranges",
        type=parse_panel_ranges,
        default=DEFAULT_PANEL_RANGE_LIMITS_KM,
        help=(
            "Comma-separated ymin-ymax ranges in km for panels a-h, "
            "e.g. '200-400,400-600,...'."
        ),
    )
    parser.add_argument(
        "--use-summary-cache",
        action="store_true",
        help="Use low-resolution cached S/N grids from simone/decoded_summaries for faster preview plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser


def plot_all_links(
    start_time=None,
    end_time=None,
    min_range_km=100.0,
    max_range_km=600.0,
    panel_ranges_km=None,
    use_summary_cache=False,
    output="rcs_all_links_fullpage.pdf",
    show=False,
    # optional time averaging (linear SN) applied before converting to dB
    time_smooth_samples=5,
    time_smooth_kernel=None,
):
    panel_labels = "abcdefgh"
    links = plot_deco.DEFAULT_RCS_LINKS
    if panel_ranges_km is None:
        panel_ranges_km = [(float(min_range_km), float(max_range_km))] * len(links)
    if len(panel_ranges_km) != len(links):
        raise ValueError("panel_ranges_km must have one (ymin, ymax) tuple per link.")

    global_vmin = -20.0
    global_vmax = None
    for (tx, rx), (panel_ymin, panel_ymax) in zip(links, panel_ranges_km):
        decoded = load_plot_grid(
            tx=tx,
            rx=rx,
            use_summary_cache=use_summary_cache,
            time_smooth_samples=time_smooth_samples if not use_summary_cache else None,
            time_smooth_kernel=time_smooth_kernel if not use_summary_cache else None,
        )
        # If we loaded a summary cache it contains dB values only; if the user
        # requested smoothing we need to operate on linear SN before converting
        # back to dB. For full decoded data, smoothing has already been applied
        # inside compute_rcs_grid when load_plot_grid forwarded the smoothing
        # parameters.
        if use_summary_cache and (time_smooth_kernel is not None or (time_smooth_samples is not None and int(time_smooth_samples) > 1)):
            # build kernel
            if time_smooth_kernel is None:
                kernel = np.repeat(1.0 / float(time_smooth_samples), int(time_smooth_samples))
            else:
                kernel = np.asarray(time_smooth_kernel, dtype=float)

            # operate on linear SN (S+N)/N
            sn_db = decoded["sn_plus_n_over_n_db"]
            sn_lin = 10.0 ** (sn_db / 10.0)
            sn_smoothed = np.empty_like(sn_lin)
            for ir in range(sn_lin.shape[0]):
                sn_smoothed[ir, :] = np.convolve(sn_lin[ir, :], kernel, mode="same")
            # convert back to dB and replace the array used for plotting
            decoded = dict(decoded)
            decoded["sn_plus_n_over_n_db"] = 10.0 * np.log10(np.maximum(sn_smoothed, 1e-12))

        _, sn_plot_db = plot_deco._slice_time_window(
            decoded["times_datetime64"],
            decoded["sn_plus_n_over_n_db"],
            start_time=start_time,
            end_time=end_time,
        )
        range_mask = (
            np.asarray(decoded["range_km"], dtype=float) >= float(panel_ymin)
        ) & (
            np.asarray(decoded["range_km"], dtype=float) <= float(panel_ymax)
        )
        if not np.any(range_mask):
            continue
        visible_values = np.asarray(sn_plot_db[range_mask, :], dtype=float)
        finite = np.isfinite(visible_values)
        if not np.any(finite):
            continue
        vmax_link = float(np.nanmax(visible_values[finite]))
        global_vmax = vmax_link if global_vmax is None else max(global_vmax, vmax_link)

    if global_vmax is None:
        raise ValueError("No finite S/N + 1 values were found in the requested plotting window.")

    print(
        f"Shared S/N + 1 color scale: vmin={global_vmin:.1f} dB, vmax={global_vmax:.2f} dB"
    )
    print(
        "Data source: "
        + ("summary cache (decimated)" if use_summary_cache else "full decoded data (native resolution)")
    )

    rcparams = dict(plot_deco.publication_rcparams())
    rcparams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    with plt.rc_context(rcparams):
        fig, axes = plt.subplots(
            4,
            2,
            figsize=(7.1, 6.0),
            sharex=True,
            sharey=False,
            constrained_layout=True,
        )
        fig.supylabel("Propagation range (km)")

        mesh = None
        for idx, (((tx, rx), (panel_ymin, panel_ymax)), ax) in enumerate(zip(zip(links, panel_ranges_km), axes.flat)):
            decoded = load_plot_grid(
                tx=tx,
                rx=rx,
                use_summary_cache=use_summary_cache,
                time_smooth_samples=time_smooth_samples if not use_summary_cache else None,
                time_smooth_kernel=time_smooth_kernel if not use_summary_cache else None,
            )
            result = plot_deco.plot_decoded(
                tx=tx,
                rx=rx,
                start_time=start_time,
                end_time=end_time,
                ymin=panel_ymin,
                ymax=panel_ymax,
                ax=ax,
                add_colorbar=False,
                show=False,
                output_filename=None,
                field_name="sn_plus_n_over_n_db",
                colorbar_label=r"$S/N + 1$ (dB)",
                vmin=global_vmin,
                vmax=global_vmax,
                precomputed_grid=decoded,
                overlay_predicted_range=True,
                show_aspect_axis=False,
            )
            if mesh is None:
                mesh = result["mesh"]

            ax.text(
                0.02,
                0.96,
                f"{panel_labels[idx]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )
            ax.grid(False)

        for row_idx in range(4):
            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                is_bottom = row_idx == 3
                ax.set_xlabel("" if not is_bottom else "Time (UTC)")
                ax.set_ylabel("")
                ax.tick_params(labelbottom=is_bottom, labelleft=True)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                if is_bottom:
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(30)
                        tick.set_horizontalalignment("right")

        cbar = fig.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            pad=0.01,
            shrink=0.98,
            aspect=40,
        )
        cbar.set_label(r"$S/N + 1$ (dB)")

        fig.savefig(output, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def main():
    args = build_parser().parse_args()
    plot_all_links(
        start_time=args.start_time,
        end_time=args.end_time,
        min_range_km=args.min_range_km,
        max_range_km=args.max_range_km,
        panel_ranges_km=args.panel_ranges,
        use_summary_cache=args.use_summary_cache,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()

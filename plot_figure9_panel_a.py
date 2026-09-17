#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import plot_fragments


PAGE_WIDTH_PT = 980.39952
PAGE_HEIGHT_PT = 353.99952


def load_histogram_data():
    _, _, fragment_ids, _, _, fragment_geo_pos, fragment_times = plot_fragments.get_fragments()
    _, _, radar_altitudes, radar_snr, _, _, _, _, _ = plot_fragments.get_radar_detections()

    optical_heights = []
    first_detection_heights = []
    for fragment_id, positions, times in zip(fragment_ids, fragment_geo_pos, fragment_times):
        heights = np.asarray(positions[:, 2], dtype=float) / 1e3
        optical_heights.extend(heights[np.isfinite(heights)])

        first_index = int(np.argmin(np.asarray(times, dtype=float)))
        first_height = float(heights[first_index])
        if np.isfinite(first_height):
            first_detection_heights.append(first_height)

    radar_heights = plot_fragments.get_radar_detection_heights_km(radar_altitudes, radar_snr)
    if len(first_detection_heights) != len(fragment_ids):
        raise ValueError(
            f"Expected one first-detection height for each of {len(fragment_ids)} fragments; "
            f"found {len(first_detection_heights)}."
        )
    return (
        np.asarray(optical_heights),
        np.asarray(radar_heights),
        np.asarray(first_detection_heights),
    )


def make_overlay(output):
    optical_heights, radar_heights, first_detection_heights = load_histogram_data()
    bins = np.arange(34.0, 88.0, 2.0)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
        }
    ):
        fig = plt.figure(figsize=(PAGE_WIDTH_PT / 72.0, PAGE_HEIGHT_PT / 72.0))
        fig.patch.set_alpha(0.0)
        fig.add_artist(
            Rectangle(
                (0.0, 0.0),
                0.272,
                1.0,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="none",
                zorder=0,
            )
        )

        ax = fig.add_axes([0.0522, 0.1966, 0.2097, 0.6802], zorder=5)
        optical_color = "#6b6b6b"
        radar_color = "#cb181d"

        ax.hist(
            optical_heights,
            bins=bins,
            orientation="horizontal",
            color=optical_color,
            alpha=0.55,
            edgecolor=optical_color,
            linewidth=0.8,
        )
        ax.set_xlim(0.0, 95.0)
        ax.set_ylim(30.0, 90.0)
        ax.set_xlabel("Number of optical detections")
        ax.set_ylabel("Altitude (km)")
        ax.tick_params(axis="x", colors=optical_color)
        ax.spines["bottom"].set_color(optical_color)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.6)

        mean_height = float(np.mean(optical_heights))
        sigma_height = float(np.std(optical_heights))
        altitude_grid = np.linspace(34.0, 86.0, 400)
        bin_width = float(bins[1] - bins[0])
        gaussian_counts = (
            optical_heights.size
            * bin_width
            / (sigma_height * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((altitude_grid - mean_height) / sigma_height) ** 2)
        )
        ax.plot(gaussian_counts, altitude_grid, "--", color="0.45", linewidth=1.2)
        ax.text(
            0.97,
            0.93,
            f"Gaussian fit\n$z_0={mean_height:.1f}$ km\n$\\sigma={sigma_height:.1f}$ km",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "0.4", "pad": 2.0},
        )
        ax.text(0.02, 0.98, "a)", transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

        ax_top = ax.twiny()
        ax_top.hist(
            radar_heights,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color=radar_color,
            linewidth=1.5,
        )
        first_counts, _ = np.histogram(first_detection_heights, bins=bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax_top.stairs(first_counts, bins, orientation="horizontal", color="black", linewidth=1.4)
        nonzero = first_counts > 0
        ax_top.plot(first_counts[nonzero], bin_centers[nonzero], "o", color="black", markersize=2.8)
        ax_top.set_xlim(0.0, 42.0)
        ax_top.set_xlabel("Radar / first detections of fragments")
        ax_top.tick_params(axis="x", colors="black")
        ax_top.spines["top"].set_color("black")
        ax_top.spines["bottom"].set_visible(False)

        handles = [
            Patch(facecolor=optical_color, edgecolor=optical_color, alpha=0.55, label="Optical"),
            Line2D([0], [0], color=radar_color, linewidth=1.5, label="Radar"),
            Line2D([0], [0], color="0.45", linestyle="--", linewidth=1.2, label="Gaussian fit"),
            Line2D([0], [0], color="black", marker="o", markersize=2.8, linewidth=1.4, label="First detections of fragments"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True)

        fig.savefig(output, transparent=True, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)

    print(f"Plotted {len(first_detection_heights)} fragment first-detection heights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    make_overlay(args.output)


if __name__ == "__main__":
    main()

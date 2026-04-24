#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt

import plot_optical_radar


DEFAULT_LABELS = ["a)", "b)", "c)", "d)"]


def parse_labels(value):
    labels = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(labels) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated panel labels.")
    return labels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a publication-quality 2x2 optical/radar figure directly from plotting functions.",
    )
    parser.add_argument(
        "--output",
        default="optical_radar_2x2.pdf",
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--labels",
        type=parse_labels,
        default=DEFAULT_LABELS,
        help="Comma-separated panel labels, e.g. 'a),b),c),d)'.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.2,
        help="Publication font scaling applied uniformly across all panels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser


def create_2x2_figure(output="optical_radar_2x2.pdf", labels=None, font_scale=1.2, show=False):
    labels = list(DEFAULT_LABELS if labels is None else labels)
    if len(labels) != 4:
        raise ValueError("labels must contain exactly four entries.")

    rcparams = plot_optical_radar.publication_rcparams(font_scale=font_scale)
    label_fontsize = 16.0 * float(font_scale)
    axis_label_fontsize = 17.0 * float(font_scale)
    tick_label_fontsize = 14.0 * float(font_scale)
    left_xlabel_fontsize = axis_label_fontsize
    right_xlabel_fontsize = axis_label_fontsize

    with plt.rc_context(rcparams):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12.8, 9.8),
            sharex="col",
            sharey=True,
            constrained_layout=True,
        )

        plot_optical_radar.plot_orbgen_time_alt_panel(
            axes[0, 0],
            title=None,
            font_scale=font_scale,
        )
        plot_optical_radar.plot_fit_relative_speed_lon_alt_panel(
            axes[0, 1],
            title=None,
            font_scale=font_scale,
            add_colorbar=True,
        )
        plot_optical_radar.plot_radar_time_alt_panel(
            axes[1, 0],
            title=None,
            font_scale=font_scale,
        )
        plot_optical_radar.plot_fit_energy_lon_alt_panel(
            axes[1, 1],
            title=None,
            font_scale=font_scale,
            add_colorbar=True,
        )

        for ax in axes.flat:
            ax.set_ylim(bottom=20.0)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=tick_label_fontsize)

        for ax in axes[0, :]:
            ax.tick_params(labelbottom=False)

        for ax in axes[:, 1]:
            ax.tick_params(labelleft=False)

        axes[1, 0].tick_params(axis="x", labelrotation=30)
        #fig.text(0.27, 0.02, "Time (UTC)", ha="center", va="center", fontsize=left_xlabel_fontsize)
        fig.text(0.74, 0.02, "Longitude (deg)", ha="center", va="center", fontsize=right_xlabel_fontsize)
        fig.supylabel("Altitude (km)", fontsize=axis_label_fontsize)

        for ax, label in zip(axes.flat, labels):
            ax.text(
                0.015,
                0.985,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_fontsize,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
                zorder=2000,
            )

        fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return output


def main():
    args = build_parser().parse_args()
    create_2x2_figure(
        output=args.output,
        labels=args.labels,
        font_scale=args.font_scale,
        show=args.show,
    )
    print(f"Saved 2x2 panel figure: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse

import plot_fragments


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot publication-quality optical and radar altitude histograms.",
    )
    parser.add_argument(
        "--output",
        default="fragment_radar_height_hist.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    plot_fragments.plot_height_hists(save_path=args.output, show=args.show)


if __name__ == "__main__":
    main()

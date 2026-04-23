#!/usr/bin/env python3

import argparse

import plot_decos
import plot_fragments
import plot_optical_radar
import plot_optical_radar_2x2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate the current Falcon 9 publication figures in one run.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive figures for the plots that support it.",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Skip fig_map_falcon9.pdf.",
    )
    parser.add_argument(
        "--skip-height-hist",
        action="store_true",
        help="Skip fragment_radar_height_hist.pdf.",
    )
    parser.add_argument(
        "--skip-optical-radar",
        action="store_true",
        help="Skip opt_radar_lon_vs_alt.pdf, opt_fit_lon_vs_alt.pdf, and opt_fit_relative_speed_lon_vs_alt.pdf.",
    )
    parser.add_argument(
        "--skip-2x2",
        action="store_true",
        help="Skip optical_radar_2x2.pdf.",
    )
    parser.add_argument(
        "--skip-snr",
        action="store_true",
        help="Skip snr_all_links_fullpage.pdf.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not args.skip_map:
        print("Generating fig_map_falcon9.pdf")
        plot_fragments.plot_map()

    if not args.skip_height_hist:
        print("Generating fragment_radar_height_hist.pdf")
        plot_fragments.plot_height_hists(
            save_path="fragment_radar_height_hist.pdf",
            show=args.show,
        )

    if not args.skip_optical_radar:
        print("Generating opt_radar_lon_vs_alt.pdf")
        plot_optical_radar.radar_plot()

        print("Generating opt_fit_lon_vs_alt.pdf")
        plot_optical_radar.fit_overlay_plot()

        print("Generating opt_fit_relative_speed_lon_vs_alt.pdf")
        plot_optical_radar.fit_overlay_velocity_plot()

    if not args.skip_2x2:
        print("Generating optical_radar_2x2.pdf")
        plot_optical_radar_2x2.create_2x2_figure(
            output="optical_radar_2x2.pdf",
            show=args.show,
        )

    if not args.skip_snr:
        print("Generating snr_all_links_fullpage.pdf")
        plot_decos.plot_all_links(
            output="snr_all_links_fullpage.pdf",
            show=args.show,
        )

    print("All requested plots completed.")


if __name__ == "__main__":
    main()

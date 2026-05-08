#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import scipy.constants
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from metablate.physics import aerodynamics
from metablate.atmosphere import AtmPymsis

PAPER_DIR = Path(__file__).resolve().parent
FALCON9_DIR = PAPER_DIR.parent / "falcon9"
HISTOGRAM_ALTITUDE_RANGE_KM = (30.0, 90.0)

if str(FALCON9_DIR) not in sys.path:
    sys.path.insert(0, str(FALCON9_DIR))

import plot_fragments  # noqa: E402

model = AtmPymsis()

print("NRL MSISE00 species:")
for name, species_data in model.species.items():
    print(f"{name}:{species_data}")

select_species = ["N2", "O2"]

@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def load_height_histogram_inputs():
    with pushd(FALCON9_DIR):
        _, _, _, _, _, fragment_geo_pos, fragment_times = plot_fragments.get_fragments()
        _, _, ralt, rsnr, _, _, _, _, _ = plot_fragments.get_radar_detections()
    # Use all fragment heights (every optical detection) instead of only the initial detection heights
    fragment_heights_km_list = []
    for geo in fragment_geo_pos:
        if geo is None or len(geo) == 0:
            continue
        # geo is an array of [lat, lon, alt_m] rows
        alts_km = np.asarray(geo[:, 2], dtype=float) / 1e3
        alts_km = alts_km[np.isfinite(alts_km)]
        if alts_km.size:
            fragment_heights_km_list.append(alts_km)
    if fragment_heights_km_list:
        fragment_heights_km = np.concatenate(fragment_heights_km_list)
    else:
        fragment_heights_km = np.asarray([], dtype=float)
    radar_heights_km = plot_fragments.get_radar_detection_heights_km(ralt, rsnr)
    return (
        np.asarray(fragment_heights_km, dtype=float),
        np.asarray(radar_heights_km, dtype=float),
    )


def load_specific_energy_loss_segments():
    fit_segments = []
    extrapolated_segments = []

    for path in sorted(FALCON9_DIR.glob("ballistic_fit_sharedstart*.h5")):
        with h5py.File(path, "r") as handle:
            for group_name, bucket in (("model", fit_segments), ("impact/trajectory", extrapolated_segments)):
                if group_name not in handle:
                    continue
                group = handle[group_name]
                times = np.asarray(group["times_model"][()], dtype=float)
                heights_km = np.asarray(group["hgt_m"][()], dtype=float) / 1e3
                lat_deg = np.asarray(group["lat_deg"][()], dtype=float)
                lon_deg = np.asarray(group["lon_deg"][()], dtype=float)
                energy_loss = np.asarray(group["specific_energy_loss_rate_w_kg"][()], dtype=float)
                speed_km_s = np.asarray(group["speed_m_s"][()], dtype=float) / 1e3

                order = np.argsort(times)
                heights_km = heights_km[order]
                energy_loss = energy_loss[order]
                mask = np.isfinite(heights_km) & np.isfinite(energy_loss) & (energy_loss > 0.0)
                if np.any(mask):
                    bucket.append(
                        {
                            "label": path.stem,
                            "height_km": heights_km[mask],
                            "energy_loss_w_kg": energy_loss[mask],
                            "lat_deg": lat_deg,
                            "lon_deg": lon_deg,
                            "times": times,
                            "speed_km_s": speed_km_s,
                        }
                    )

    if not fit_segments:
        raise FileNotFoundError(
            f"No usable specific-energy-loss data were found in {FALCON9_DIR}/ballistic_fit_sharedstart*.h5"
        )

    return fit_segments, extrapolated_segments


def load_speed_segments():
    fit_segments = []
    extrapolated_segments = []

    for path in sorted(FALCON9_DIR.glob("ballistic_fit_sharedstart*.h5")):
        with h5py.File(path, "r") as handle:
            for group_name, bucket in (("model", fit_segments), ("impact/trajectory", extrapolated_segments)):
                if group_name not in handle:
                    continue
                group = handle[group_name]
                times = np.asarray(group["times_model"][()], dtype=float)
                heights_km = np.asarray(group["hgt_m"][()], dtype=float) / 1e3
                speed_km_s = np.asarray(group["speed_m_s"][()], dtype=float) / 1e3

                order = np.argsort(times)
                heights_km = heights_km[order]
                speed_km_s = speed_km_s[order]
                mask = np.isfinite(heights_km) & np.isfinite(speed_km_s) & (speed_km_s > 0.0)
                if np.any(mask):
                    bucket.append(
                        {
                            "label": path.stem,
                            "height_km": heights_km[mask],
                            "speed_km_s": speed_km_s[mask],
                        }
                    )

    if not fit_segments:
        raise FileNotFoundError(
            f"No usable speed data were found in {FALCON9_DIR}/ballistic_fit_sharedstart*.h5"
        )

    return fit_segments, extrapolated_segments


def load_dynamic_pressure_segments():
    fit_segments = []
    extrapolated_segments = []

    for path in sorted(FALCON9_DIR.glob("ballistic_fit_sharedstart*.h5")):
        with h5py.File(path, "r") as handle:
            for group_name, bucket in (("model", fit_segments), ("impact/trajectory", extrapolated_segments)):
                if group_name not in handle:
                    continue
                group = handle[group_name]
                times = np.asarray(group["times_model"][()], dtype=float)
                heights_km = np.asarray(group["hgt_m"][()], dtype=float) / 1e3
                rho_a = np.asarray(group["rho_a_kg_m3"][()], dtype=float)
                relative_speed = np.asarray(group["relative_speed_m_s"][()], dtype=float)
                dynamic_pressure_pa = 0.5 * rho_a * relative_speed**2

                order = np.argsort(times)
                heights_km = heights_km[order]
                dynamic_pressure_pa = dynamic_pressure_pa[order]
                mask = (
                    np.isfinite(heights_km)
                    & np.isfinite(dynamic_pressure_pa)
                    & (dynamic_pressure_pa > 0.0)
                )
                if np.any(mask):
                    bucket.append(
                        {
                            "label": path.stem,
                            "height_km": heights_km[mask],
                            "dynamic_pressure_pa": dynamic_pressure_pa[mask],
                        }
                    )

    if not fit_segments:
        raise FileNotFoundError(
            f"No usable dynamic-pressure data were found in {FALCON9_DIR}/ballistic_fit_sharedstart*.h5"
        )

    return fit_segments, extrapolated_segments


def compute_histogram_bins(fragment_initial_heights_km, radar_heights_km, bin_width_km=2.0):
    all_heights = np.concatenate((fragment_initial_heights_km, radar_heights_km))
    all_heights = all_heights[np.isfinite(all_heights)]
    if all_heights.size == 0:
        raise ValueError("No valid optical or radar heights are available.")

    hmin = bin_width_km * np.floor(np.min(all_heights) / bin_width_km)
    hmax = bin_width_km * np.ceil(np.max(all_heights) / bin_width_km)
    bins = np.arange(hmin, hmax + bin_width_km, bin_width_km)
    if bins.size < 2:
        bins = np.array([hmin, hmin + bin_width_km], dtype=float)
    return bins, float(hmin), float(hmax)


def compute_energy_limits(fit_segments, extrapolated_segments):
    positive_values = []
    for segment in fit_segments + extrapolated_segments:
        values = np.asarray(segment["energy_loss_w_kg"], dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            positive_values.append(values)

    if not positive_values:
        return 1.0, 10.0

    all_values = np.concatenate(positive_values)
    lo = float(np.nanpercentile(all_values, 2.0))
    hi = float(np.nanpercentile(all_values, 98.0))
    if not np.isfinite(lo) or lo <= 0.0:
        lo = float(np.nanmin(all_values[all_values > 0.0]))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(all_values))

    vmin = 10.0 ** np.floor(np.log10(lo))
    vmax = 10.0 ** np.ceil(np.log10(hi))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def compute_speed_limits(fit_segments, extrapolated_segments):
    values = []
    for segment in fit_segments + extrapolated_segments:
        speed = np.asarray(segment["speed_km_s"], dtype=float)
        speed = speed[np.isfinite(speed) & (speed > 0.0)]
        if speed.size:
            values.append(speed)

    if not values:
        return 1.0, 10.0

    all_values = np.concatenate(values)
    vmin = float(np.nanpercentile(all_values, 2.0))
    vmax = float(np.nanpercentile(all_values, 98.0))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(all_values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(all_values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def compute_dynamic_pressure_limits(fit_segments, extrapolated_segments):
    positive_values = []
    for segment in fit_segments + extrapolated_segments:
        values = np.asarray(segment["dynamic_pressure_pa"], dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            positive_values.append(values)

    if not positive_values:
        return 1e2, 1e5

    all_values = np.concatenate(positive_values)
    lo = float(np.nanpercentile(all_values, 2.0))
    hi = float(np.nanpercentile(all_values, 98.0))
    if not np.isfinite(lo) or lo <= 0.0:
        lo = float(np.nanmin(all_values[all_values > 0.0]))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(all_values))

    vmin = 10.0 ** np.floor(np.log10(lo))
    vmax = 10.0 ** np.ceil(np.log10(hi))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def calculate_shock_data(segments):
    for data in segments:
        atm = model.density(
            time=np.array(data["times"][0], dtype="datetime64[s]"),
            lat=data["lat_deg"][:1],
            lon=data["lon_deg"][:1],
            alt=data["height_km"]*1e3,
            mass_densities=False,
        )
        temp = atm["Temperature"].values.flatten()
        num_tot = np.zeros_like(temp)
        mean_mass = np.zeros_like(temp)
        for symbol in select_species:
            num_tot += atm[symbol].values.flatten()
        mean_mass = atm["Total"].values.flatten() / num_tot

        sound_speeds = aerodynamics.speed_of_sound_air(temp, mean_mass)
        mach_numbers = data["speed_km_s"] * 1e3 / sound_speeds
        post_shock_temps = aerodynamics.rankine_hugoniot_post_shock_temperature(
            temp, mach_numbers
        )
        eff_area = np.pi * (3.7e-10 / 2)**2
        mfp = aerodynamics.atmospheric_mean_free_path(num_tot, eff_area)
        Kn = mfp / 1.0
        mach_numbers[Kn > 0.005] = np.nan
        post_shock_temps[Kn > 0.005] = np.nan

        data["post_shock_T"] = post_shock_temps


def make_figure(output_path: Path, show=False):
    fragment_initial_heights_km, radar_heights_km = load_height_histogram_inputs()
    fit_segments, extrapolated_segments = load_specific_energy_loss_segments()
    speed_fit_segments, speed_extrapolated_segments = load_speed_segments()
    dynamic_pressure_fit_segments, dynamic_pressure_extrapolated_segments = load_dynamic_pressure_segments()
    bins, hmin, hmax = compute_histogram_bins(fragment_initial_heights_km, radar_heights_km)
    energy_vmin, energy_vmax = compute_energy_limits(fit_segments, extrapolated_segments)
    speed_vmin, speed_vmax = compute_speed_limits(speed_fit_segments, speed_extrapolated_segments)

    calculate_shock_data(fit_segments)
    calculate_shock_data(extrapolated_segments)

    optical_color = "#6b6b6b"
    radar_color = "#cb181d"
    fit_color = "black"
    extrapolated_color = "black"

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 0.9,
        }
    ):
        fig, (ax_hist, ax_energy, ax_speed, ax_temp) = plt.subplots(
            1,
            4,
            figsize=(13.6, 4.8),
            sharey=True,
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.0, 1.15, 0.95, 1.05)},
        )

        panel_label_style = {
            "ha": "left",
            "va": "top",
            "fontsize": 16,
            "fontweight": "bold",
        }

        ax_hist.hist(
            fragment_initial_heights_km,
            bins=bins,
            orientation="horizontal",
            color=optical_color,
            alpha=0.55,
            edgecolor=optical_color,
            linewidth=0.8,
        )
        ax_hist.set_xlabel("Number of optical detections")
        ax_hist.set_ylabel("Altitude (km)")
        ax_hist.tick_params(axis="x", colors=optical_color)
        ax_hist.spines["bottom"].set_color(optical_color)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.grid(axis="y", color="0.88", linewidth=0.8)
        ax_hist.text(0.02, 0.98, "a)", transform=ax_hist.transAxes, **panel_label_style)

        ax_hist_top = ax_hist.twiny()
        ax_hist_top.hist(
            radar_heights_km,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color=radar_color,
            linewidth=1.8,
        )
        ax_hist_top.set_xlabel("Number of radar detections")
        ax_hist_top.tick_params(axis="x", colors=radar_color)
        ax_hist_top.spines["top"].set_color(radar_color)
        ax_hist_top.spines["bottom"].set_visible(False)
        ax_hist.set_ylim(*HISTOGRAM_ALTITUDE_RANGE_KM)

        hist_handles = [
            Patch(
                facecolor=optical_color,
                edgecolor=optical_color,
                alpha=0.55,
                label="Optical",
            ),
            Line2D([0], [0], color=radar_color, linewidth=1.8, label="Radar"),
        ]
        ax_hist.legend(handles=hist_handles, frameon=True, loc="lower right")

        for segment in extrapolated_segments:
            ax_energy.semilogx(
                segment["energy_loss_w_kg"],
                segment["height_km"],
                linestyle="--",
                color=extrapolated_color,
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )
        for segment in fit_segments:
            ax_energy.semilogx(
                segment["energy_loss_w_kg"],
                segment["height_km"],
                linestyle="-",
                color=fit_color,
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )

        ax_energy.set_xlim(3e3, energy_vmax)
        ax_energy.set_ylim(*HISTOGRAM_ALTITUDE_RANGE_KM)
        ax_energy.set_xlabel("Specific energy loss rate\n" + r"(W kg$^{-1}$)")
        ax_energy.tick_params(axis="y", labelleft=False)
        ax_energy.grid(axis="y", which="both", linestyle="--", linewidth=0.5, color="0.86")
        ax_energy.text(0.02, 0.98, "b)", transform=ax_energy.transAxes, **panel_label_style)

        energy_handles = [
            Line2D([0], [0], color=fit_color, linewidth=1.4, label="Fit"),
            Line2D([0], [0], color=extrapolated_color, linestyle="--", linewidth=1.1, label="Extrapolated"),
        ]
        ax_energy.legend(handles=energy_handles, frameon=True, loc="upper right")

        for segment in speed_extrapolated_segments:
            ax_speed.plot(
                segment["speed_km_s"],
                segment["height_km"],
                linestyle="--",
                color=extrapolated_color,
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )
        for segment in speed_fit_segments:
            ax_speed.plot(
                segment["speed_km_s"],
                segment["height_km"],
                linestyle="-",
                color=fit_color,
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )

        ax_speed.set_xlim(3, 8)
        ax_speed.set_ylim(*HISTOGRAM_ALTITUDE_RANGE_KM)
        ax_speed.set_xlabel("Velocity\n" + r"(km s$^{-1}$)")
        ax_speed.tick_params(axis="y", labelleft=False)
        ax_speed.grid(axis="y", linestyle="--", linewidth=0.5, color="0.86")
        ax_speed.text(0.02, 0.98, "c)", transform=ax_speed.transAxes, **panel_label_style)

        for segment in extrapolated_segments:
            ax_temp.plot(
                segment["post_shock_T"],
                segment["height_km"],
                linestyle="--",
                color=extrapolated_color,
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )
        for segment in fit_segments:
            ax_temp.plot(
                segment["post_shock_T"],
                segment["height_km"],
                linestyle="-",
                color=fit_color,
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )

        ax_temp.set_ylim(*HISTOGRAM_ALTITUDE_RANGE_KM)
        ax_temp.set_xlabel("Post-shock temperature\n(K)")
        ax_temp.tick_params(axis="y", labelleft=False)
        ax_temp.grid(axis="y", which="both", linestyle="--", linewidth=0.5, color="0.86")
        ax_temp.text(0.02, 0.98, "d)", transform=ax_temp.transAxes, **panel_label_style)
        print("saving to %s"%(output_path))
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication figure with the fragment/radar detection-height histogram, "
            "specific kinetic-energy-loss, speed, and dynamic-pressure profiles from ballistic_fit_sharedstart*.h5."
        )
    )
    parser.add_argument(
        "--output",
        default="fragment_radar_height_hist_3col.pdf",
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
    make_figure(output_path=Path(args.output).resolve(), show=args.show)


if __name__ == "__main__":
    main()

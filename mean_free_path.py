#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymsis import msis


DEFAULT_EFFECTIVE_COLLISION_DIAMETER_M = 3.7e-10


def run_msis_state(times_dt64, lat_deg, lon_deg, alt_km):
    times_dt64 = np.asarray(times_dt64, dtype="datetime64[s]")
    lat_deg = np.asarray(lat_deg, dtype=float)
    lon_deg = np.asarray(lon_deg, dtype=float)
    alt_km = np.asarray(alt_km, dtype=float)
    return np.asarray(
        msis.run(
            times_dt64,
            lat_deg,
            lon_deg,
            alt_km,
            geomagnetic_activity=-1,
        )
    )


def extract_total_number_density_m3(msis_output):
    arr = np.asarray(msis_output, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("Unexpected scalar MSIS output.")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 3:
        raise ValueError("Unexpected MSIS output shape for species number densities.")
    return np.nansum(arr[..., 1:-1], axis=-1)


def mean_free_path_from_number_density(
    number_density_m3,
    effective_collision_diameter_m=DEFAULT_EFFECTIVE_COLLISION_DIAMETER_M,
):
    number_density_m3 = np.asarray(number_density_m3, dtype=float)
    sigma_m2 = np.pi * effective_collision_diameter_m**2
    return 1.0 / (
        np.sqrt(2.0) * sigma_m2 * np.maximum(number_density_m3, 1e-30)
    )


def mean_free_path_m(
    time_dt64,
    lat_deg,
    lon_deg,
    alt_km,
    effective_collision_diameter_m=DEFAULT_EFFECTIVE_COLLISION_DIAMETER_M,
):
    alt_km = np.asarray(alt_km, dtype=float)
    scalar_input = alt_km.ndim == 0
    alt_km = np.atleast_1d(alt_km)
    times_dt64 = np.full(alt_km.shape, np.datetime64(time_dt64, "s"), dtype="datetime64[s]")
    lat_deg = np.full(alt_km.shape, float(lat_deg), dtype=float)
    lon_deg = np.full(alt_km.shape, float(lon_deg), dtype=float)

    msis_output = run_msis_state(times_dt64, lat_deg, lon_deg, alt_km)
    total_number_density_m3 = extract_total_number_density_m3(msis_output)
    lambda_m = mean_free_path_from_number_density(
        total_number_density_m3,
        effective_collision_diameter_m=effective_collision_diameter_m,
    )
    if scalar_input:
        return float(lambda_m[0])
    return lambda_m


def build_mean_free_path_profile(
    time_dt64,
    lat_deg,
    lon_deg,
    altitude_grid_m,
    effective_collision_diameter_m=DEFAULT_EFFECTIVE_COLLISION_DIAMETER_M,
):
    """
    Evaluate the MSIS mean free path on a provided altitude grid in meters.
    """
    altitude_grid_m = np.asarray(altitude_grid_m, dtype=float)
    mean_free_path_profile_m = mean_free_path_m(
        time_dt64=time_dt64,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_km=altitude_grid_m / 1e3,
        effective_collision_diameter_m=effective_collision_diameter_m,
    )
    return {
        "time_dt64": np.datetime64(time_dt64, "s"),
        "lat_deg": float(lat_deg),
        "lon_deg": float(lon_deg),
        "altitude_grid_m": altitude_grid_m,
        "mean_free_path_m": np.asarray(mean_free_path_profile_m, dtype=float),
        "effective_collision_diameter_m": float(effective_collision_diameter_m),
    }


def build_mean_free_path_profile_from_density_profile(
    density_profile,
    effective_collision_diameter_m=DEFAULT_EFFECTIVE_COLLISION_DIAMETER_M,
):
    """
    Reuse a fit_ballistic3 density-profile reference state to compute a mean
    free path profile on the same altitude grid.
    """
    altitude_grid_m = np.asarray(density_profile["altitude_grid_m"], dtype=float)
    reference_time_dt64 = np.datetime64(
        int(round(float(density_profile["reference_time_unix"]))),
        "s",
    )
    return build_mean_free_path_profile(
        time_dt64=reference_time_dt64,
        lat_deg=float(density_profile["reference_lat_deg"]),
        lon_deg=float(density_profile["reference_lon_deg"]),
        altitude_grid_m=altitude_grid_m,
        effective_collision_diameter_m=effective_collision_diameter_m,
    )


def altitude_for_mean_free_path_match(
    altitude_grid_m,
    mean_free_path_profile_m,
    target_length_m,
):
    """
    Interpolate the altitude where the mean free path equals a target length.

    Returns `None` if no crossing exists on the supplied grid.
    """
    altitude_grid_m = np.asarray(altitude_grid_m, dtype=float)
    mean_free_path_profile_m = np.asarray(mean_free_path_profile_m, dtype=float)
    target_length_m = float(target_length_m)

    valid = (
        np.isfinite(altitude_grid_m)
        & np.isfinite(mean_free_path_profile_m)
        & (mean_free_path_profile_m > 0.0)
    )
    altitude_grid_m = altitude_grid_m[valid]
    mean_free_path_profile_m = mean_free_path_profile_m[valid]
    if altitude_grid_m.size < 2 or target_length_m <= 0.0:
        return None

    log_ratio = np.log(mean_free_path_profile_m / target_length_m)
    exact = np.where(np.isclose(log_ratio, 0.0, atol=1e-12))[0]
    if exact.size > 0:
        return float(altitude_grid_m[exact[0]])

    crossing_idx = np.where(log_ratio[:-1] * log_ratio[1:] < 0.0)[0]
    if crossing_idx.size == 0:
        return None

    i0 = int(crossing_idx[0])
    i1 = i0 + 1
    x0 = float(log_ratio[i0])
    x1 = float(log_ratio[i1])
    if abs(x1 - x0) < 1e-12:
        return float(0.5 * (altitude_grid_m[i0] + altitude_grid_m[i1]))
    alpha = -x0 / (x1 - x0)
    return float(altitude_grid_m[i0] + alpha * (altitude_grid_m[i1] - altitude_grid_m[i0]))


def altitude_for_mean_free_path_match_in_interval(
    altitude_grid_m,
    mean_free_path_profile_m,
    target_length_m,
    min_altitude_m=None,
    max_altitude_m=None,
):
    """
    Return the crossing altitude only if it lies inside the requested interval.
    """
    match_altitude_m = altitude_for_mean_free_path_match(
        altitude_grid_m=altitude_grid_m,
        mean_free_path_profile_m=mean_free_path_profile_m,
        target_length_m=target_length_m,
    )
    if match_altitude_m is None:
        return None
    if min_altitude_m is not None and match_altitude_m < float(min_altitude_m):
        return None
    if max_altitude_m is not None and match_altitude_m > float(max_altitude_m):
        return None
    return match_altitude_m


def format_mean_free_path(lambda_m):
    lambda_m = float(lambda_m)
    if lambda_m < 1e-3:
        return f"{lambda_m * 1e6:.1f} um"
    if lambda_m < 1.0:
        return f"{lambda_m * 1e3:.2f} mm"
    return f"{lambda_m:.2f} m"


def plot_mean_free_path_profile(
    time_dt64=np.datetime64("2025-02-19T03:46:05"),
    lat_deg=54.0,
    lon_deg=12.0,
    min_alt_km=30.0,
    max_alt_km=120.0,
    n_alt=400,
    output="mean_free_path_profile.pdf",
    show=True,
):
    alt_km = np.linspace(float(min_alt_km), float(max_alt_km), int(n_alt))
    profile = build_mean_free_path_profile(
        time_dt64=time_dt64,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        altitude_grid_m=alt_km * 1e3,
    )
    lambda_m = profile["mean_free_path_m"]

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    ax.semilogx(lambda_m, alt_km, color="black", linewidth=1.3)
    ax.set_xlabel("Mean free path (m)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("MSIS mean free path profile", pad=4)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.text(
        0.98,
        0.02,
        f"{np.datetime_as_string(np.datetime64(time_dt64, 's'))}\n{lat_deg:.1f}$^\\circ$N, {lon_deg:.1f}$^\\circ$E",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 2.0},
    )

    fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "alt_km": alt_km,
        "mean_free_path_m": lambda_m,
        "figure": fig,
        "axes": ax,
        "output": str(Path(output)),
    }


if __name__ == "__main__":
    plot_mean_free_path_profile()

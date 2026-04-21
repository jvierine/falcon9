from datetime import datetime, timezone
from pathlib import Path
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

from era5wind import ERA5Wind


BASE_DIR = Path(__file__).resolve().parent
RADAR_DIR = BASE_DIR / "radar"
OUTPUT_DIR = BASE_DIR / "plots"
ERA5_CACHE = BASE_DIR / "data" / "era5_germany_20250219_03_04.nc"
RADAR_SNR_THRESHOLD_DB = 20.0
GERMANY_ERA5_AREA = [56.5, 4.0, 47.0, 16.5]  # north, west, south, east
PLOT_TOP_KM = 90.0
N_HEIGHT_SAMPLES = 361


def get_reference_point():
    lats = []
    lons = []
    hgts = []
    times = []
    snrs = []

    for radar_file in sorted(glob.glob(str(RADAR_DIR / "*.h5"))):
        with h5py.File(radar_file, "r") as handle:
            lats.append(np.asarray(handle["latitude"][()]))
            lons.append(np.asarray(handle["longitude"][()]))
            hgts.append(np.asarray(handle["altitude_m"][()]))
            times.append(np.asarray(handle["time_unix"][()]))
            snrs.append(np.asarray(handle["peak_power_db"][()]))

    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    hgt = np.concatenate(hgts)
    t = np.concatenate(times)
    snr = np.concatenate(snrs)

    good = snr > RADAR_SNR_THRESHOLD_DB
    if not np.any(good):
        raise ValueError("No radar detections above the requested SNR threshold")

    return {
        "lat_deg": float(np.median(lat[good])),
        "lon_deg": float(np.median(lon[good])),
        "hgt_m": float(np.median(hgt[good])),
        "time_unix": float(np.median(t[good])),
        "count": int(np.count_nonzero(good)),
    }


def main():
    ref = get_reference_point()

    era5 = ERA5Wind.download(
        target_path=ERA5_CACHE,
        start_time_unix=ref["time_unix"] - 3600.0,
        end_time_unix=ref["time_unix"] + 3600.0,
        area=GERMANY_ERA5_AREA,
    )

    hgt_profile_m, _, _ = era5.profile_enu(
        ref["lat_deg"],
        ref["lon_deg"],
        unix_time=ref["time_unix"],
    )
    hgt_m = np.linspace(0.0, PLOT_TOP_KM * 1e3, N_HEIGHT_SAMPLES)
    u_east = np.zeros_like(hgt_m)
    v_north = np.zeros_like(hgt_m)
    for idx, hgt_now_m in enumerate(hgt_m):
        u_east[idx], v_north[idx], _ = era5.wind_enu(
            ref["lat_deg"],
            ref["lon_deg"],
            hgt_now_m,
            unix_time=ref["time_unix"],
        )
    speed = np.sqrt(u_east**2 + v_north**2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "era5_wind_profile_germany_falcon9.png"

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.5), constrained_layout=True, sharey=True)

    axes[0].plot(u_east, hgt_m / 1e3, label="u east", color="C0", lw=2)
    axes[0].plot(v_north, hgt_m / 1e3, label="v north", color="C1", lw=2)
    axes[0].axhline(ref["hgt_m"] / 1e3, color="0.4", ls="--", lw=1.2, label="Median radar altitude")
    axes[0].set_xlabel("Wind component (m/s)")
    axes[0].set_ylabel("Height (km)")
    axes[0].set_title("ERA5 Components")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.0, PLOT_TOP_KM)

    axes[1].plot(speed, hgt_m / 1e3, color="C2", lw=2, label="Horizontal speed")
    axes[1].axhline(ref["hgt_m"] / 1e3, color="0.4", ls="--", lw=1.2)
    axes[1].set_xlabel("Horizontal wind speed (m/s)")
    axes[1].set_title("ERA5 Speed")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_ylim(0.0, PLOT_TOP_KM)

    event_dt = datetime.fromtimestamp(ref["time_unix"], tz=timezone.utc)
    fig.suptitle(
        (
            "ERA5 wind profile over Germany during Falcon 9 re-entry\n"
            f"{event_dt:%Y-%m-%d %H:%M:%S} UTC at "
            f"{ref['lat_deg']:.2f} N, {ref['lon_deg']:.2f} E "
            f"(median of {ref['count']} radar detections)"
        ),
        fontsize=11,
    )

    note = (
        f"ERA5 pressure-level top reaches {np.max(hgt_profile_m)/1e3:.1f} km; "
        f"winds are set to zero above that height; "
        f"median radar altitude was {ref['hgt_m']/1e3:.1f} km."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)

    fig.savefig(out_path, dpi=180)
    print(out_path)
    print(event_dt.isoformat())
    print(ref)
    print(f"ERA5 native profile top_km={np.max(hgt_profile_m)/1e3:.2f}")


if __name__ == "__main__":
    main()

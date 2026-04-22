#!/usr/bin/env python3
"""
Compute the antipodal point of a reference location.

By default, this script uses the Lambert Centre as the "centre of Australia":
25°36'36.4"S, 134°21'17.3"E.

Geoscience Australia notes that Australia does not have one single official
centre, so an alternate "centre of gravity" point is also included.
"""

from __future__ import annotations

import argparse
import math


AUSTRALIA_CENTERS = {
    "lambert": {
        "name": "Lambert Centre of Australia",
        "lat_deg": -(25.0 + 36.0 / 60.0 + 36.4 / 3600.0),
        "lon_deg": 134.0 + 21.0 / 60.0 + 17.3 / 3600.0,
    },
    "gravity": {
        "name": "Australia centre-of-gravity point",
        "lat_deg": -(23.0 + 7.0 / 60.0),
        "lon_deg": 132.0 + 8.0 / 60.0,
    },
}


def normalize_lon(lon_deg: float) -> float:
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0


def antipode(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    anti_lat = -float(lat_deg)
    anti_lon = normalize_lon(float(lon_deg) + 180.0)
    return anti_lat, anti_lon


def deg_to_dms(value_deg: float, positive_hemisphere: str, negative_hemisphere: str) -> str:
    hemisphere = positive_hemisphere if value_deg >= 0.0 else negative_hemisphere
    value_abs = abs(float(value_deg))
    degrees = int(value_abs)
    minutes_full = (value_abs - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    return f"{degrees:02d}°{minutes:02d}'{seconds:05.2f}\"{hemisphere}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the antipodal location of the centre of Australia.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(AUSTRALIA_CENTERS.keys()),
        default="lambert",
        help="Reference point for Australia's centre.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Override latitude in decimal degrees.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Override longitude in decimal degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference = AUSTRALIA_CENTERS[args.method]
    lat_deg = reference["lat_deg"] if args.lat is None else float(args.lat)
    lon_deg = reference["lon_deg"] if args.lon is None else normalize_lon(args.lon)
    anti_lat_deg, anti_lon_deg = antipode(lat_deg, lon_deg)

    print(f"Reference point: {reference['name']}")
    print(f"Latitude (deg):  {lat_deg:.6f}")
    print(f"Longitude (deg): {lon_deg:.6f}")
    print(
        "Reference point (DMS): "
        f"{deg_to_dms(lat_deg, 'N', 'S')}, "
        f"{deg_to_dms(lon_deg, 'E', 'W')}"
    )
    print()
    print("Antipodal point:")
    print(f"Latitude (deg):  {anti_lat_deg:.6f}")
    print(f"Longitude (deg): {anti_lon_deg:.6f}")
    print(
        "Antipodal point (DMS): "
        f"{deg_to_dms(anti_lat_deg, 'N', 'S')}, "
        f"{deg_to_dms(anti_lon_deg, 'E', 'W')}"
    )


if __name__ == "__main__":
    main()

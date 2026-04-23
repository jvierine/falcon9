#!/usr/bin/env python3

from pathlib import Path
from pprint import pprint
import h5py
import json

DATA_DIR = Path("simone/decoded_files")


def decode_name(value):
    """
    Convert HDF5 string/bytes dataset content into a normal Python string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()

    # Handle scalar numpy bytes/strings or 1-element arrays
    try:
        if hasattr(value, "shape") and value.shape == ():
            value = value.item()
            if isinstance(value, bytes):
                return value.decode("utf-8").strip()
            return str(value).strip()
    except Exception:
        pass

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        if isinstance(value, bytes):
            return value.decode("utf-8").strip()
        return str(value).strip()

    return str(value).strip()


def read_name(h5, path):
    return decode_name(h5[path][()])


def read_latlon(h5, path):
    """
    Read [lat, lon, alt] and return (lat, lon).
    """
    data = h5[path][()]
    return (float(data[0]), float(data[1]))


def extract_station_coordinates(data_dir: Path):
    coords = {
        "tx": {},
        "rx": {},
    }

    for filepath in sorted(data_dir.glob("*.h5")):
        try:
            with h5py.File(filepath, "r") as h5:
                tx_name = read_name(h5, "/metadata/system/tx/name")
                rx_name = read_name(h5, "/metadata/system/rx/name")

                tx_latlon = read_latlon(h5, "/metadata/system/tx/gps")
                rx_latlon = read_latlon(h5, "/metadata/system/rx/gps")

        except Exception as e:
            print(f"Skipping {filepath.name}: {e}")
            continue

        # Store TX coordinates, warn if inconsistent
        if tx_name in coords["tx"]:
            if coords["tx"][tx_name] != tx_latlon:
                print(
                    f"Warning: TX station '{tx_name}' has inconsistent coordinates: "
                    f"{coords['tx'][tx_name]} vs {tx_latlon} in {filepath.name}"
                )
        else:
            coords["tx"][tx_name] = tx_latlon

        # Store RX coordinates, warn if inconsistent
        if rx_name in coords["rx"]:
            if coords["rx"][rx_name] != rx_latlon:
                print(
                    f"Warning: RX station '{rx_name}' has inconsistent coordinates: "
                    f"{coords['rx'][rx_name]} vs {rx_latlon} in {filepath.name}"
                )
        else:
            coords["rx"][rx_name] = rx_latlon

    return coords


if __name__ == "__main__":
    station_coords = extract_station_coordinates(DATA_DIR)

    print("\nPython dictionary:")
    pprint(station_coords)

    print("\nJSON:")
    print(json.dumps(station_coords, indent=2))
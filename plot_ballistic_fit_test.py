#!/usr/bin/env python3

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import glob
import numpy as n
    

def main():
    fit_files = glob.glob("ballistic_fit_shared*.h5")
    if not fit_files:
        raise SystemExit("No ballistic fit files found.")

    for path in fit_files:
        with h5py.File(path, "r") as h5:
            if "impact" not in h5 or "trajectory" not in h5["impact"]:
                continue

            group = h5["impact/trajectory"]
            lon_deg = group["lon_deg"][()]
            hgt = group["hgt_m"][()]
            #print(hgt)
            specific_energy_loss = group["specific_energy_loss_rate_w_kg"][()]
            print(specific_energy_loss)

        plt.scatter(lon_deg, hgt, c=specific_energy_loss, s=8, alpha=0.8, label=path)

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Hgt")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

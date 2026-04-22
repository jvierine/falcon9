import h5py
import os
import numpy as n

directory = "SIMONe_geodetic_Falcon_2025"
fl = glob = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".h5")]
fl.sort()

if fl:
    f = fl[0]
    print(f"Checking {f}")
    with h5py.File(f, "r") as h:
        print("Attributes:")
        for k in h.attrs.keys():
            print(f"{k}: {h.attrs[k]}")
        print("\nKeys:")
        for k in h.keys():
            if isinstance(h[k], h5py.Dataset):
                print(f"{k}: {h[k].shape}")
            else:
                print(f"{k}: Group")
        if "doppler_hz" in h:
            print(f"Sample doppler_hz: {h['doppler_hz'][:5]}")

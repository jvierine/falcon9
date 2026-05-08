import numpy as np
import matplotlib.pyplot as plt
import h5py

from metablate.physics import aerodynamics
from metablate.atmosphere import AtmPymsis

model = AtmPymsis()

print("NRL MSISE00 species:")
for name, species_data in model.species.items():
    print(f"{name}:{species_data}")

select_species = ["O", "N2", "O2"]

bases = ["/model", "/impact/trajectory"]
vel = []
alts = []
lat_deg = []
lon_deg = []

with h5py.File("ballistic_fit_sharedstart_1.h5", "r") as hf:
    for base in bases:
        vel.append(hf[f"{base}/relative_speed_m_s"][()])
        alts.append(hf[f"{base}/hgt_m"][()])
        lat_deg.append(hf[f"{base}/lat_deg"][()])
        lon_deg.append(hf[f"{base}/lon_deg"][()])
    times_model = np.array([hf["model/times_model"][0]], dtype="datetime64[s]")

vel = np.concat(vel)
alts = np.concat(alts)
lat_deg = np.concat(lat_deg)
lon_deg = np.concat(lon_deg)

atm = model.density(
    time=times_model,
    lat=lat_deg[:1],
    lon=lon_deg[:1],
    alt=alts,
    mass_densities=True,
)
temp = atm["Temperature"].values.flatten()
num_tot = atm["Total"].values.flatten() / model.mean_mass


sound_speeds = aerodynamics.speed_of_sound_air(temp, model.mean_mass)
mach_numbers = aerodynamics.mach_number(vel, sound_speeds)
post_shock_temps = aerodynamics.rankine_hugoniot_post_shock_temperature(
    temp, mach_numbers
)
eff_area = np.pi * (3.7e-10 / 2)**2
mfp = aerodynamics.atmospheric_mean_free_path(num_tot, eff_area)
Kn = mfp / 1.0

mach_numbers[Kn > 0.005] = np.nan
post_shock_temps[Kn > 0.005] = np.nan

fig, axes = plt.subplots(1, 3, sharey=True)

axes[0].plot(mach_numbers, alts * 1e-3)
axes[1].plot(post_shock_temps, alts * 1e-3)
axes[2].plot(Kn, alts * 1e-3)

plt.show()

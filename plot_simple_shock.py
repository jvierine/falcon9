import numpy as np
import matplotlib.pyplot as plt
import h5py

from metablate.physics import aerodynamics
from metablate.atmosphere import AtmPymsis

model = AtmPymsis()

print("NRL MSISE00 species:")
for name, species_data in model.species.items():
    print(f"{name}:{species_data}")

select_species = ["N2", "O2"]

bases = ["/model", "/impact/trajectory"]
vel_ = []
alts_ = []
lat_deg_ = []
lon_deg_ = []

with h5py.File("ballistic_fit_sharedstart_1.h5", "r") as hf:
    for base in bases:
        vel_.append(hf[f"{base}/relative_speed_m_s"][()])
        alts_.append(hf[f"{base}/hgt_m"][()])
        lat_deg_.append(hf[f"{base}/lat_deg"][()])
        lon_deg_.append(hf[f"{base}/lon_deg"][()])
    times_model = np.array([hf["model/times_model"][0]], dtype="datetime64[s]")

vel = np.concat(vel_)
alts = np.concat(alts_)
lat_deg = np.concat(lat_deg_)
lon_deg = np.concat(lon_deg_)

atm = model.density(
    time=times_model,
    lat=lat_deg[:1],
    lon=lon_deg[:1],
    alt=alts,
    mass_densities=False,
)
temp = atm["Temperature"].values.flatten()
num_tot = np.zeros_like(temp)
mean_mass = np.zeros_like(temp)
for symbol in select_species:
    num_tot += atm[symbol].values.flatten()
mean_mass = atm["Total"].values.flatten() / num_tot

sound_speeds = aerodynamics.speed_of_sound_air(temp, mean_mass)
mach_numbers = vel / sound_speeds
post_shock_temps = aerodynamics.rankine_hugoniot_post_shock_temperature(
    temp, mach_numbers
)

min_mach = 4 * np.sqrt(1/2 - 1/(2 * 1.4))
post_shock_mach_nums = aerodynamics.rankine_hugoniot_post_shock_mach_number(
    mach_numbers
)
post_shock_mach_nums[mach_numbers < min_mach] = np.nan
post_shock_speeds = post_shock_mach_nums * sound_speeds


eff_area = np.pi * (3.7e-10 / 2)**2
mfp = aerodynamics.atmospheric_mean_free_path(num_tot, eff_area)
Kn = mfp / 1.0

mach_numbers[Kn > 0.005] = np.nan
post_shock_temps[Kn > 0.005] = np.nan

fig, axes = plt.subplots(1, 4, sharey=True)

axes[0].plot(mach_numbers, alts * 1e-3)
axes[1].plot(post_shock_mach_nums, alts * 1e-3)

axes[2].plot(post_shock_temps, alts * 1e-3)
axes[3].plot(Kn, alts * 1e-3)

fig, axes = plt.subplots(1, 2, sharey=True)

axes[0].plot(vel, alts * 1e-3)
axes[1].plot(post_shock_speeds, alts * 1e-3)

plt.show()

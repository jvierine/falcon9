"""
Calculate total atmospheric mass between 50-70 km using MSIS model via pymsis.
"""
import numpy as n
import numpy as np
from scipy.integrate import trapezoid
import pymsis
import matplotlib.pyplot as plt

# Create a range of dates during the 2003 Halloween storm
dates = np.arange(
    np.datetime64("2003-10-28T00:00"), 
    np.datetime64("2003-11-04T00:00"), 
    np.timedelta64(30, "m")
)

# Define altitude range for integration (50-70 km)
altitudes = np.linspace(50, 70, 50)  # 50 points for smooth integration

# Location: equator
lon, lat = 0, 0

print("Calculating integrated atmospheric mass between 50-70 km")
print("=" * 70)
print(f"Location: ({lon}°, {lat}°)")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Altitude range: {altitudes[0]:.0f} - {altitudes[-1]:.0f} km")
print("=" * 70)

# Run the model for all altitudes and dates
# geomagnetic_activity=-1 is a storm-time run
# Output shape: [ndates, nlons, nlats, nalts, 11]
# Index [0] is Total Mass Density (kg/m³)
data = pymsis.calculate(
    dates=dates,
    lons=lon,
    lats=lat,
    alts=altitudes,
    geomagnetic_activity=-1  # Storm-time run
)

# Extract total mass density for all dates and altitudes
# data shape: [ndates, 1, 1, nalts, 11]
# We want: [ndates, nalts]
density = data[:, 0, 0, :, 0]  # Total mass density in kg/m³

# Integrate over altitude to get column mass (kg/m²)
# For each date, integrate rho(z) dz over 50-70 km
column_mass = np.zeros(len(dates))

R_earth=6380e3
m_total=0
dalt=altitudes[1] - altitudes[0]
for i in range(len(altitudes)):
    m_total += density[0,i]*dalt*4*n.pi*(R_earth+altitudes[i]*1e3)**2

print(f"Total atmospheric mass between 50-70 km: {m_total:.2e} kg or %1.2f tons."%(m_total/1e3))
print("The annual spacecraft material influx is %g parts per million of the atmospheric mass."%(1e6*0.89e6/m_total))

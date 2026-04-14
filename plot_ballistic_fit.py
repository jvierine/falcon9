import h5py 
import matplotlib.pyplot as plt
import numpy as n
h=h5py.File("ballistic_fit_2.h5","r")


t=h["model/times_model"][()]
vr=h["model/relative_speed_m_s"][()]
hgt=h["model/hgt_m"][()]
lon=h["model/lon_deg"][()]
elr=h["model/specific_energy_loss_rate_w_kg"][()]
rho_a=h["model/rho_a_kg_m3"][()]



it=h["impact/trajectory/times_model"][()]
ivr=h["impact/trajectory/relative_speed_m_s"][()]
ihgt=h["impact/trajectory/hgt_m"][()]
ilon=h["impact/trajectory/lon_deg"][()]
ielr=h["impact/trajectory/specific_energy_loss_rate_w_kg"][()]
irho_a=h["impact/trajectory/rho_a_kg_m3"][()]

plt.scatter(lon,hgt,c=n.log10(0.5*rho_a*(vr**2)),vmin=2,vmax=4.5)
plt.scatter(ilon,ihgt,c=n.log10(0.5*irho_a*(ivr**2)),vmin=2,vmax=4.5)

plt.colorbar()
plt.show()


plt.scatter(lon,hgt,c=vr,vmin=0,vmax=7.6e3)
plt.scatter(ilon,ihgt,c=ivr,vmin=0,vmax=7.6e3)
plt.colorbar()
plt.show()

plt.scatter(lon,hgt,c=n.log10(elr),vmin=2,vmax=6)
plt.scatter(ilon,ihgt,c=n.log10(ielr),vmin=2,vmax=6)
plt.colorbar()
plt.show()
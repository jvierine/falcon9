import pymsis 
import numpy as n


dates=[n.datetime64("2024-10-28T00:00")]
lat=0
lon=0

r=n.linspace(40,80,num=200)

data=pymsis.calculate(dates[0],lat,lon,r)
print(data.shape)

rho_a=data[0,0,0,:,0]

dr=n.diff(r)[0]*1e3
m_total=0
R_earth=6380e3
for i in range(len(r)):
    m_total += rho_a[i]*dr*4*n.pi*(R_earth+r[i]*1e3)**2
print(m_total)

m_spacecraft_influx=1600e3 # kg per year (Schultz et.al., 2026)
print("influx %1.0f tons per day"%(m_spacecraft_influx/365/1e3))

print("annual space waste mass / atmospheric mass (40-80 km) %1.3f ppt"%(1e12*m_spacecraft_influx/m_total))

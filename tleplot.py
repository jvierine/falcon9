import numpy as np
import matplotlib.pyplot as plt
import sgp4
from sgp4.api import Satrec, jday, days2mdhms
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone
import numpy as n

radius_earth_km=6378.135
f=open("data/tles.txt","r")
a=[]
jds=[]
l1s=[]
l2s=[]
while True:
    l1=f.readline()
    if l1 == "":
        break
    l2=f.readline()
    l1s.append(l1)
    l2s.append(l2)
    satellite = Satrec.twoline2rv(l1, l2)
    a_km = radius_earth_km*satellite.a
    a.append(a_km)
    year = satellite.epochyr
    day_of_year = satellite.epochdays

    jd=satellite.jdsatepoch+satellite.jdsatepochF
    jds.append(jd)

    
JD_UNIX_EPOCH = 2440587.5
jds=np.array(jds)
seconds_since_epoch = (jds - JD_UNIX_EPOCH) * 86400
datetime_unix = seconds_since_epoch.astype('timedelta64[s]') + np.datetime64('1970-01-01T00:00:00')
a=np.array(a)
da=np.diff(a)/np.diff(jds)

plt.plot(datetime_unix,a-radius_earth_km,".")
plt.ylabel("Semi-major axis - Earth radius (km)")
plt.xlabel("Time (UTC)")
plt.title("Falcon 9 upper stage\n(NORAD ID:62878)")
plt.show()


ts = load.timescale()
s = EarthSatellite(l1s[-1], l2s[-1], "F9", ts)

dt = datetime(2025, 2, 19, 3, 0, 0, tzinfo=timezone.utc)
t0 = ts.from_datetime(dt)
jd0_utc = t0.tt

jd_times = n.linspace(jd0_utc,jd0_utc+47*60/(86400),num=1000)
t=ts.tt_jd(jd_times)
geocentric=s.at(t)
subpoints=geocentric.subpoint()
lats=subpoints.latitude.degrees
longs=subpoints.longitude.degrees
hgt=subpoints.elevation.km*1e3


plt.plot(longs,lats,".")
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")

plt.show()

plt.plot(jd_times,hgt/1e3,".")
plt.xlabel("Time (JD)")
plt.ylabel("Height (km)")

plt.show()

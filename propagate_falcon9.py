import numpy as np
import matplotlib.pyplot as plt
import sgp4
from sgp4.api import Satrec, jday, days2mdhms
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone
import numpy as n
from skyfield.framelib import itrs
from pymsis import msis
import scipy.constants as c
from skyfield.api import wgs84

import plot_fragments
import jcoord


def propagate(pos0,vel0,tmax=3*3600,dt=10,A_to_m=1e-3):

    pos=[]
    ts=[]
    vels=[]
    hgts=[]
    lats=[]
    lons=[]
    
    n_t = int(tmax/dt)
    pos_now=pos0*1e3
    vel_now=vel0*1e3
    
    for i in range(n_t):

        msis_dates = n.array([n.datetime64("2025-02-19T03:00")])
        radius=n.linalg.norm(pos_now)
        hgt=radius-radius_earth_km*1e3
        if hgt < 0:
            continue
#        print(hgt)
        # msis assumes utc date, lon (deg), lat (deg), hgt (km above sea level)
        data=msis.run(msis_dates, -5, 40.0, hgt/1e3, geomagnetic_activity=-1)
        print(data.shape)
        rho_a=data[0][0]
        

        gamma=0.5
        v2=n.linalg.norm(vel_now)**2.0
        # TBD calculate the true gravitational acceleration based on height
        M_earth=5.9722e24
        
        g0=c.G*M_earth/(radius)**2.0
        print(g0)
        g = -g0*pos_now/n.linalg.norm(pos_now)
        v_unit = vel_now/n.linalg.norm(vel_now)

        print("atmospheric density %1.2g kg/m^3 hgt %1.2f km vel %1.2f km/s"%(rho_a,hgt/1e3,n.sqrt(v2)/1e3))
        # add gravitational field perturbations!
        
        # add forces together
        dv_dt = -gamma*A_to_m*rho_a*v2*v_unit + g
        # update velocity
        dv = dv_dt*dt
        print(dv)
        vel_next = vel_now + dv
        # update position
        pos_next = pos_now + (0.5*(vel_now + vel_next))*dt

        pos.append(pos_now)
        vels.append(vel_now)
        hgts.append(hgt)
        lat,lon,el=itrs2latlonh(pos_now[0],pos_now[1],pos_now[2])
        lats.append(lat)
        lons.append(lon)
        vel_now=vel_next
        pos_now=pos_next
    
    return(pos,vels,hgts,lats,lons)
        
        

def compare_fragments_with_propagation():
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=plot_fragments.get_fragments()

    def itrs2latlonh(x,y,z):
        llh=jcoord.ecef2geodetic(x,y,z)
        return(llh[0],llh[1],llh[2])

    radius_earth_km=6378.135
    f=open("data/tles.txt","r")
    a=[]
    jds=[]
    l1s=[]
    l2s=[]

    # Read all TLEs that are obtained from space-track.org
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

        # these are all the TLE epochs
        jd=satellite.jdsatepoch+satellite.jdsatepochF
        jds.append(jd)

    # 1st of jan, 1970 in JD
    JD_UNIX_EPOCH = 2440587.5
    jds=np.array(jds)
    seconds_since_epoch = (jds - JD_UNIX_EPOCH) * 86400
    datetime_unix = seconds_since_epoch.astype('timedelta64[s]') + np.datetime64('1970-01-01T00:00:00')
    a=np.array(a)
    da=np.diff(a)/np.diff(jds)

    if False:
        plt.plot(datetime_unix,a-radius_earth_km,".")
        plt.ylabel("Semi-major axis - Earth radius (km)")
        plt.xlabel("Time (UTC)")
        plt.title("Falcon 9 upper stage\n(NORAD ID:62878)")
        plt.show()


    ts = load.timescale()
    # last known position
    tle_num=-4
    s = EarthSatellite(l1s[tle_num], l2s[tle_num], "F9", ts)
    t=ts.tt_jd(jds[tle_num])
    geocentric=s.at(t)
    # ECEF position and velocity
    # used as initial condition for numerical propagation

    subpoints=geocentric.subpoint()
    lats=subpoints.latitude.degrees
    longs=subpoints.longitude.degrees
    hgt=subpoints.elevation.km*1e3
    plt.plot([longs],[lats],"x")

    posvel=geocentric.frame_xyz_and_velocity(itrs)
    pos0=posvel[0].km
    vel0=posvel[1].km_per_s


    pos,vels,hgts,lats,lons=propagate(pos0,vel0,tmax=3600,dt=10,A_to_m=1e-3)
    plt.plot(lons,lats,".")

    for fp in range(len(fragment_pos)):
        plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,0])
    plt.show()


compare_fragments_with_propagation()

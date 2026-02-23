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

radius_earth_km=6378.135

def itrs2latlonh(x,y,z):
    llh=jcoord.ecef2geodetic(x,y,z)
    return(llh[0],llh[1],llh[2])

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
    tnow=0.0
    for i in range(n_t):

        msis_dates = n.array([n.datetime64("2025-02-19T03:00")])
        radius=n.linalg.norm(pos_now)
        
        llh=jcoord.ecef2geodetic(pos_now[0],pos_now[1],pos_now[2])
        hgt=llh[2]
        if hgt < 0:
            continue
        
        # msis assumes utc date, lon (deg), lat (deg), hgt (km above sea level)
        data=msis.run(msis_dates, -5, 40.0, hgt/1e3, geomagnetic_activity=-1)
        print(data.shape)
        rho_a=data[0][0]
        

        gamma=0.5
        v2=n.linalg.norm(vel_now)**2.0
        # TBD calculate the true gravitational acceleration based on height
        M_earth=5.9722e24
        
        g0=c.G*M_earth/(radius)**2.0

        g = -g0*pos_now/n.linalg.norm(pos_now)
        v_unit = vel_now/n.linalg.norm(vel_now)

        print("atmospheric density %1.2g kg/m^3 hgt %1.2f km vel %1.2f km/s"%(rho_a,hgt/1e3,n.sqrt(v2)/1e3))
        # add gravitational field perturbations!
        
        # add forces together
        dv_dt = -gamma*A_to_m*rho_a*v2*v_unit + g
        # update velocity
        dv = dv_dt*dt

        vel_next = vel_now + dv
        # update position
        pos_next = pos_now + (0.5*(vel_now + vel_next))*dt

        pos.append(pos_now)
        vels.append(vel_now)
        hgts.append(hgt)
        lat,lon,el=itrs2latlonh(pos_now[0],pos_now[1],pos_now[2])
        lats.append(lat)
        lons.append(lon)
        tnow+=dt
        ts.append(tnow)
        vel_now=vel_next
        pos_now=pos_next
    
    return(pos,vels,hgts,lats,lons,ts)
        
        

def compare_fragments_with_propagation():
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=plot_fragments.get_fragments()
    
    import pandas as pd
    
    cols = [
        "timestamp", "dt_min", "IFA", "F10_7", "FB10_7", "Ap", "IDW",
        "rho_kg_m3", "Vn_kms", "Ve_kms", "MMWT", "Tloc_K", "Texo_K",
        "LST_h", "GLat_deg", "GLon_deg", "GAlt_km", "Va_kms", "gam_deg",
        "gload", "qdot_W_m2", "SRat", "TRat", "KnInf", "MaInf",
        "CD", "CD_CD0", "Orb", "ULat_deg", "dS_km",
        "Hpe_km", "Hap_km", "H_km", "Torb_min", "mjd1950_d"
    ]
    
    # ---- Read the file ----
    df = pd.read_csv(
        "data/prediction.dat",
        comment="#",          # ignore OrbGen header/comments
        delim_whitespace=True,# split on arbitrary whitespace
        names=cols,
        engine="python"
    )
    lats=df["GLat_deg"][0:2]
    lons=df["GLon_deg"][0:2]
    alts=df["GAlt_km"][0:2]
    print(lats)
    print(lons)
    print(alts)
    x,y,z=jcoord.geodetic2ecef(lats,lons,alts*1e3)

    vel0=n.array([x[1]-x[0],y[1]-y[0],z[1]-z[0]])/60.0
    pos0=n.array([x[0],y[0],z[0]])
    llh=jcoord.ecef2geodetic(x[0],y[0],z[0])
    print(llh)
#    exit(0)
#    print(df)
    

 #   pos0=posvel[0].km
 #   vel0=posvel[1].km_per_s


    pos,vels,hgts,lats,lons,ts=propagate(pos0/1e3,vel0/1e3,tmax=3600,dt=1,A_to_m=1e-3)
    plt.plot(lons,lats,".")

    for fp in range(len(fragment_pos)):
        plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,0])

    plt.plot(df["GLon_deg"][:],df["GLat_deg"][:],".")
    plt.show()

    for fp in range(len(fragment_pos)):
        if fp==0:
            plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,2]/1e3,".",label="Observation")
        plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,2]/1e3,".")
            
    plt.plot(df["GLon_deg"][:],df["GAlt_km"][:],label="ESA prediction")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Height (km)")
    plt.legend()
    plt.show()
    
    

#    plt.plot(ts,hgts,".")
 #   plt.show()


compare_fragments_with_propagation()

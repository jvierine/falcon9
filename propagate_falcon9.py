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
import scipy.interpolate as sint
import scipy.optimize as so
import plot_fragments
import jcoord
import h5py
radius_earth_km=6378.135

def itrs2latlonh(x,y,z):
    llh=jcoord.ecef2geodetic(x,y,z)
    return(llh[0],llh[1],llh[2])

def propagate(pos0,
              vel0,
              tmax=3*3600,
              dt=1,
              A_to_m=1e-3,
              msis_date0=n.datetime64("2025-02-19T03:30")):
    """
    pos0 ITRS initial position
    vel0 ITRS initial velocity
    tmax maximum propagation time
    dt integration step
    A_to_m area to mass ratio on the object.
    """
    pos=[]
    ts=[]
    vels=[]
    hgts=[]
    lats=[]
    lons=[]
    rhos=[]
    n_t = int(tmax/dt)
    pos_now=pos0
    vel_now=vel0
    tnow=0.0
    for i in range(n_t):

        msis_dates = n.array([msis_date0])
        radius=n.linalg.norm(pos_now)
        
        llh=jcoord.ecef2geodetic(pos_now[0],pos_now[1],pos_now[2])
        hgt=llh[2]
#        if hgt < 0:
 #           continue
        
        # msis assumes utc date, lon (deg), lat (deg), hgt (km above sea level)
        try:
            data=msis.run(msis_dates, -5, 40.0, hgt/1e3, geomagnetic_activity=-1)
            #        print(data.shape)
            rho_a=data[0][0]
        except:
            print("problem with msis")
            rho_a=1.0

        

        gamma=0.5
        v2=n.linalg.norm(vel_now)**2.0
        # TBD calculate the true gravitational acceleration based on height
        M_earth=5.9722e24
        
        g0=c.G*M_earth/(radius)**2.0

        g = -g0*pos_now/n.linalg.norm(pos_now)
        v_unit = vel_now/n.linalg.norm(vel_now)

#        print("atmospheric density %1.2g kg/m^3 hgt %1.2f km vel %1.2f km/s"%(rho_a,hgt/1e3,n.sqrt(v2)/1e3))
        # add gravitational field perturbations!
        
        # add forces together
        dv_dt = -gamma*A_to_m*rho_a*v2*v_unit + g
        # update velocity
        dv = dv_dt*dt

        vel_next = vel_now + dv
        # update position
        pos_next = pos_now + (0.5*(vel_now + vel_next))*dt
        rhos.append(rho_a)
        pos.append(pos_now)
        vels.append(n.linalg.norm(vel_now))
        hgts.append(hgt)
        lat,lon,el=itrs2latlonh(pos_now[0],pos_now[1],pos_now[2])
        lats.append(lat)
        lons.append(lon)
        tnow+=dt
        ts.append(tnow)
        vel_now=vel_next
        pos_now=pos_next
    
    return(pos,vels,hgts,lats,lons,ts,rhos)
        
        

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


    pos,vels,hgts,lats,lons,ts,rhos=propagate(pos0/1e3,vel0/1e3,tmax=3600,dt=1,A_to_m=1e-3)
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
    
def initial_guess(t,x,y,z):
    n_m=len(t)
    A=n.zeros([n_m,2])
    t0=n.min(t)
    A[:,0]=1
    A[:,1]=t-t0
    x0,vx=n.linalg.lstsq(A,x)[0]
    y0,vy=n.linalg.lstsq(A,y)[0]
    z0,vz=n.linalg.lstsq(A,z)[0]
    return(n.array([x0,y0,z0]),n.array([vx,vy,vz]))

def forward_model(x,t_max):
    p0=x[0:3]
    v0=x[3:6]
    AtM=x[6]
  #  print("p0",p0)
 #   print("v0",v0)
#    print("atm",AtM)
    m_pos,m_vels,m_hgts,m_lats,m_lons,m_ts,m_rhos=propagate(p0,v0,
                                                            tmax=t_max,
                                                            dt=0.2,
                                                            A_to_m=AtM,
                                                            msis_date0=n.datetime64("2025-02-19T03:30"))
    m_pos=n.array(m_pos)
    m_ts=n.array(m_ts)
    
    m_ts[0]=m_ts[0]-1.0
    fx=sint.interp1d(m_ts,m_pos[:,0])
    fy=sint.interp1d(m_ts,m_pos[:,1])
    fz=sint.interp1d(m_ts,m_pos[:,2])
    frho=sint.interp1d(m_ts,m_rhos)
    fvel=sint.interp1d(m_ts,m_vels)

    return(fx,fy,fz,frho,fvel)

#dm_pos0,dm_vel0,dm_A_to_m=fit_drag_model(t_this,x_this,y_this,z_this,pos0,vel0)
def fit_drag_model(t_this,x_this,y_this,z_this,pos0,vel0,fname="fit.png"):
    t_max=(n.max(t_this)-n.min(t_this))+1.0
    t0=n.min(t_this)
    def ss(x):
        fx,fy,fz,frho,fvel=forward_model(x,t_max)
        model_x=fx(t_this-t0)
        model_y=fy(t_this-t0)
        model_z=fz(t_this-t0)
        s=n.sum( (model_x-x_this)**2.0+(model_y-y_this)**2.0+(model_z-z_this)**2.0)
    #    print(s,x)

        return(s)
    x_guess=n.zeros(7)
    x_guess[0:3]=pos0
    x_guess[3:6]=vel0
    x_guess[6]=1e-3
    
    xhat=so.fmin(ss,x_guess)

    fx,fy,fz,frho,fvel=forward_model(xhat,t_max)
    tmodel=n.linspace(0,t_max-1,num=100)

    if True:
        plt.figure()
        plt.subplot(131)
        plt.plot(tmodel,fx(tmodel))
        plt.plot(t_this-t0,x_this,"x")
        
        plt.subplot(132)
        plt.plot(tmodel,fy(tmodel))
        plt.plot(t_this-t0,y_this,"x")
        
        plt.subplot(133)
        plt.plot(tmodel,fz(tmodel))
        plt.plot(t_this-t0,z_this,"x")
        plt.savefig(fname)
        plt.close()
    return(xhat[0:3],xhat[3:6],xhat[6],fx(t_this-t0),fy(t_this-t0),fz(t_this-t0),frho(t_this-t0),fvel(t_this-t0))

        

    
    
    
    
    
def fit_observed_trajectory(max_duration=20):
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=plot_fragments.get_fragments()

    # go through all fragments
    for i in range(len(fragment_ids)):
        print(i)

        # chop trajectory into shorter sub trajectories if too long
        t0=n.min(fragment_times[i])
        t1=n.max(fragment_times[i])
        n_blocks=int(n.ceil((t1-t0)/max_duration))
        for bi in range(n_blocks):
            bidx=n.where( ( fragment_times[i] > (t0+bi*max_duration)) & ( fragment_times[i] < (t0+(bi+1)*max_duration)) )[0]
            if len(bidx) < 10:
                continue
            print(bidx)

            t_this=fragment_times[i][bidx]
            x_this=fragment_pos[i][bidx,0]
            y_this=fragment_pos[i][bidx,1]
            z_this=fragment_pos[i][bidx,2]
            t0=n.min(t_this)
            pos0,vel0=initial_guess(t_this,x_this,y_this,z_this)
            vmag=(n.linalg.norm(vel0)/1e3)
            if False:
                plt.subplot(131)
                plt.plot([t_this[0]],[pos0[0]],"x")
                plt.plot(t_this,x_this,".")

                plt.plot(t_this,pos0[0]+vel0[0]*(t_this-t_this[0]))

                plt.subplot(132)
                plt.plot([t_this[0]],[pos0[1]],"x")
                plt.plot(t_this,y_this,".")

                plt.plot(t_this,pos0[1]+vel0[1]*(t_this-t_this[0]))

                plt.subplot(133)
                plt.plot([t_this[0]],[pos0[2]],"x")
                plt.plot(t_this,z_this,".")

                plt.plot(t_this,pos0[2]+vel0[2]*(t_this-t_this[0]))
                plt.show()
            dm_pos0,dm_vel0,dm_A_to_m,fitx,fity,fitz,fitrho,fitvel=fit_drag_model(t_this,x_this,y_this,z_this,pos0,vel0,fname="plots/%s_%1.2f_fit.png"%(fragment_ids[i],t0))
            print("best fit ",fragment_ids[i],t0,dm_pos0,dm_vel0,dm_A_to_m)
            ho=h5py.File("fits/%s_%1.2f.h5"%(fragment_ids[i],t0),"w")
            ho["t0"]=t0
            ho["id"]=fragment_ids[i]
            ho["t_this"]=t_this
            ho["x_this"]=x_this
            ho["y_this"]=y_this
            ho["z_this"]=z_this
            ho["fitx"]=fitx
            ho["fity"]=fity
            ho["fitz"]=fitz
            ho["fitrho"]=fitrho
            ho["fitvel"]=fitvel

            ho["drag_fit_p0"]=dm_pos0
            ho["drag_fit_v0"]=dm_vel0
            ho["drag_fit_A_to_m"]=dm_A_to_m
            ho.close()


            
#    plt.colorbar()#plt.scatter(t_this,pos0[0]+vel0[0]*(t_this-t_this[0]))
 #   plt.show()
    
    
    
if __name__ == "__main__":
    fit_observed_trajectory()
    #compare_fragments_with_propagation()

import h5py
import numpy as n 
import matplotlib.pyplot as plt
import glob
import jcoord
import plot_fragments as pfs


def plot_fits():
    fl=glob.glob("fits/*.h5")
    for f in fl:
        h=h5py.File(f,"r")
        p0=h["drag_fit_p0"][()]
        atm=h["drag_fit_A_to_m"][()]
        px=h["fitx"][()]
        py=h["fity"][()]
        pz=h["fitz"][()]
        rho=h["fitrho"][()]
        vel=h["fitvel"][()]
        vel0=h["drag_fit_v0"][()]        
        t_this=h["t_this"][()]
        drag_energy=vel**3*rho*atm
        atms=n.repeat(atm,len(t_this))
        llhs=[]
        for i in range(len(px)):
            llhs.append(jcoord.ecef2geodetic(px[i],py[i],pz[i]))
        t0=h["t0"][()]
        llhs=n.array(llhs)
        plt.scatter(t_this,llhs[:,2]/1e3,c=n.log10(rho*atms),vmin=-7,vmax=-4.5)
        h.close()
    cb=plt.colorbar()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Altitude (km)")
    # 
    cb.set_label(r"$llg10(\rho_a A/m)$")#Dynamic pressure (N m$^{-2}$)")

    plt.show()


    fl=glob.glob("fits/*.h5")
    for f in fl:
        h=h5py.File(f,"r")
        p0=h["drag_fit_p0"][()]
        atm=h["drag_fit_A_to_m"][()]
        px=h["fitx"][()]
        py=h["fity"][()]
        pz=h["fitz"][()]
        rho=h["fitrho"][()]
        vel=h["fitvel"][()]
        vel0=h["drag_fit_v0"][()]        
        t_this=h["t_this"][()]
        drag_energy=vel**3*rho*atm
        llhs=[]
        for i in range(len(px)):
            llhs.append(jcoord.ecef2geodetic(px[i],py[i],pz[i]))
        t0=h["t0"][()]
        llhs=n.array(llhs)
        plt.scatter(pfs.unix_to_datetime(t_this),llhs[:,2]/1e3,c=n.log10(drag_energy),vmin=4,vmax=6)
        h.close()
    cb=plt.colorbar()
    cb.set_label(r"Energy loss rate $v^3 \rho_a A/m$ (log10 J s$^{-1}$ kg$^{-1}$)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Altitude (km)")
    plt.grid()
    plt.show()

    fl=glob.glob("fits/*.h5")
    for f in fl:
        h=h5py.File(f,"r")
        p0=h["drag_fit_p0"][()]
        atm=h["drag_fit_A_to_m"][()]
        px=h["fitx"][()]
        py=h["fity"][()]
        pz=h["fitz"][()]
        rho=h["fitrho"][()]
        vel=h["fitvel"][()]
        vel0=h["drag_fit_v0"][()]        
        t_this=h["t_this"][()]
        drag_energy=vel**3*rho*atm
        llhs=[]
        for i in range(len(px)):
            llhs.append(jcoord.ecef2geodetic(px[i],py[i],pz[i]))
        t0=h["t0"][()]
        llhs=n.array(llhs)
        plt.scatter(pfs.unix_to_datetime(t_this),llhs[:,2]/1e3,c=n.log10((vel**2.0)*atm),vmin=4,vmax=6)
        plt.xlabel("Time")
        h.close()
    cb=plt.colorbar()
    # 
    cb.set_label(r"Dynamic pressure (N m$^{-2}$)")

    plt.show()

plot_fits()
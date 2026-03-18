import numpy as n
import h5py
import glob 
import re
import jcoord
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ---- Convert unix seconds to datetime ----
def unix_to_datetime(t):
    return [datetime.utcfromtimestamp(tt) for tt in t]

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

def get_fragments():
    fl=glob.glob("fragments/*AMS*.h5")
    fl.sort()

    fragment_ids=[]
 #   fragment_files=[]
    for f in fl:
        m=re.match(r"fragments/(.*)_AMS.*_AMS.*\.h5",f)
        if m:
            fid=str(m.group(1))
            if fid not in fragment_ids:
                fragment_ids.append(fid)

#    print(fragment_ids)
#    exit(0)

    fragment_pos=[]
    fragment_pos_err=[]

    fragment_geo_pos=[]
    fragment_times=[]
    for fid in fragment_ids:
        geo_pos=[]
        pos=[]
        pos_err=[]

        times=[]
        fl=glob.glob("fragments/%s_*.h5"%(fid))
        fl.sort()
        for f in fl:
#            print(f)
            h=h5py.File(f,"r")
            tpos=h["pos_est"][()]
            tposerr=h["pos_err"][()]

            pos.append(tpos)
            pos_err.append(tposerr)
            times.append(h["time"][()])
            llh=jcoord.ecef2geodetic(tpos[0],tpos[1],tpos[2])
            geo_pos.append(llh)
            h.close()
        fragment_pos.append(n.array(pos))
        fragment_geo_pos.append(n.array(geo_pos))
        fragment_pos_err.append(n.array(pos_err))
#        fragment_files.append(f)
        fragment_times.append(n.array(times))#,"s"))
    #fig,(ax1,ax2)=plt.subplots(2,1)

    # go through all fragment id
    hgt_count=n.zeros(120)
    hgt_count_all=n.zeros(120)

    for fp in range(len(fragment_pos)):
        geo=fragment_geo_pos[fp]
        hgt_count[:]=0
        hgt_count[n.array(n.round(geo[:,2]/1e3),dtype=int)]=1
        hgt_count_all+=hgt_count
    return(hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times)

def get_radar_detections():
    fl=glob.glob("radar/*.h5")
    alts=[]
    lats=[]
    lons=[]
    snrs=[]
    times=[]
    for f in fl:
        h=h5py.File(f,"r")
        alts.append(h["altitude_m"][()])
        lats.append(h["latitude"][()])
        lons.append(h["longitude"][()])
        snrs.append(h["peak_power_db"][()])
        times.append(h["time_unix"][()])
        h.close()
    return(lats,lons,alts,snrs,times)


if __name__ == "__main__":
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()
    rlat,rlon,ralt,rsnr,rtime=get_radar_detections()
    rthresh=20



        # Single-column journal width (~3.4–3.6 inches typical)
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 6.0),
                            sharex=True,
                            constrained_layout=True)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)

    # =========================================================
    # (a) Latitude vs Time
    # =========================================================
    ax = axs[0]

    for i in range(len(rtime)):
        ridx = n.where(rsnr[i] > rthresh)[0]
        ax.plot(unix_to_datetime(rtime[i][ridx]),
                rlat[i][ridx],
                "x", alpha=0.2, color="black")

    for i in range(len(fragment_times)):
        ax.plot(unix_to_datetime(fragment_times[i]),
                fragment_geo_pos[i][:, 0], ".",color="C0")#, label="%s"%(fragment_ids[i]))
#    ax.legend()
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("(a)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # =========================================================
    # (b) Longitude vs Time
    # =========================================================
    ax = axs[1]

    for i in range(len(rtime)):
        ridx = n.where(rsnr[i] > rthresh)[0]
        ax.plot(unix_to_datetime(rtime[i][ridx]),
                rlon[i][ridx],
                "x", alpha=0.2, color="black")

    for i in range(len(fragment_times)):
        ax.plot(unix_to_datetime(fragment_times[i]),
                fragment_geo_pos[i][:, 1],
                ".", color="C0")

    ax.set_ylabel("Longitude (deg)")
    ax.set_title("(b)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # =========================================================
    # (c) Altitude vs Time
    # =========================================================
    ax = axs[2]

    for i in range(len(rtime)):
        ridx = n.where(rsnr[i] > rthresh)[0]
        if i == 0:
            ax.plot(unix_to_datetime(rtime[i][ridx]),
                    ralt[i][ridx] / 1e3,
                    "x", alpha=0.2, color="black", label="Radar")
        else:
            ax.plot(unix_to_datetime(rtime[i][ridx]),
                    ralt[i][ridx] / 1e3,
                    "x", alpha=0.2, color="black")

    for i in range(len(fragment_times)):
        if i == 0:
            ax.plot(unix_to_datetime(fragment_times[i]),
                    fragment_geo_pos[i][:, 2] / 1e3,
                    ".", color="C0", label="Fragment")
        else:
            ax.plot(unix_to_datetime(fragment_times[i]),
                    fragment_geo_pos[i][:, 2] / 1e3,
                    ".", color="C0")

    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(frameon=False)
    ax.set_title("(c)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.show()
    radar_alts=[]
    for i in range(len(ralt)):
        radar_alts=n.concatenate((radar_alts,ralt[i]/1e3))

    rcounts,rbins=n.histogram(radar_alts,bins=20)
    fig, ax1 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    # Camera (bottom x-axis)
    line1, = ax1.plot(
        hgt_count_all,
        n.arange(120),
        "s",
        ms=4,
        color="C0",
        label="Camera",
    )

    ax1.set_xlabel("Number of fragments (camera)")
    ax1.set_ylabel("Height (km)")

    # Radar (top x-axis)
    ax2 = ax1.twiny()

    radar_height = 0.5 * (rbins[:-1] + rbins[1:])
    line2, = ax2.plot(
        rcounts,
        radar_height,
        marker="o",
        color="C1",
        linestyle="none",
        ms=3,
        label="Radar",
    )

    ax2.set_xlabel("Number of fragments (radar)")

    # Combine legend cleanly
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")

    plt.show()


    # plot lat vs long with cartopy on map of europe
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature  
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':') 
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    for i in range(len(rtime)):
        ridx=n.where(rsnr[i]>rthresh)[0]
        if i==0:
            plt.plot(rlon[i][ridx],rlat[i][ridx],"x",alpha=0.2,color="black",label="radar")
        else:
            plt.plot(rlon[i][ridx],rlat[i][ridx],"x",alpha=0.2,color="black")

    for fp in range(len(fragment_pos)):
        if fp==0:
            plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,0],".",label="fragment %s"%(fragment_ids[fp]))
        else:
            plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,0],".")
    plt.legend()

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Fragment ground tracks")
    plt.legend()        
    # add gridlines
    gl = ax.gridlines(draw_labels=True)
    plt.show()

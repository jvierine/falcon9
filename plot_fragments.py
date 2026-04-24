import numpy as n
import h5py
import glob 
import re
import jcoord
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
     #       print(h.keys())
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
    bragg_enu=[]
    dops=[]
    rxs=[]
    txs=[]
    for f in fl:
        h=h5py.File(f,"r")
        m = re.search(r"geodetic_data_([^_]+)_([^_]+)_\d{8}_corr\.h5$", f)
        tx = m.group(1)
        rx = m.group(2)

        txs.append(tx)
        rxs.append(rx)


        #print(h.keys())
        alts.append(h["altitude_m"][()])
        lats.append(h["latitude"][()])
        bragg_enu.append(h["bragg_enu"][()])
        lons.append(h["longitude"][()])
        snrs.append(h["peak_power_db"][()])
        times.append(h["time_unix"][()])
        dops.append(h["doppler_hz"][()])

        h.close()
    return(lats,lons,alts,snrs,times,bragg_enu,dops,txs,rxs)


def get_predicted_impacts():
    impact_points = []
    fit_files = sorted(glob.glob("ballistic_fit_shared*.h5"))
    for fit_file in fit_files:
        try:
            with h5py.File(fit_file, "r") as h:
                if "impact" not in h:
                    continue
                impact_group = h["impact"]
                if "impact_lat_deg" not in impact_group or "impact_lon_deg" not in impact_group:
                    continue
                lat = float(n.asarray(impact_group["impact_lat_deg"][()]))
                lon = float(n.asarray(impact_group["impact_lon_deg"][()]))
                if n.isfinite(lat) and n.isfinite(lon):
                    impact_points.append((lat, lon, os.path.basename(fit_file)))
        except Exception:
            continue
    return impact_points


def polyfit_pos(t,y,deg = 5):
    # data
    t = n.asarray(t, dtype=float)
    y = n.asarray(y, dtype=float)

    # center time
    t_mean = t.mean()
    tc = t - t_mean

    # polynomial degree

    # fit
    coeffs = n.polyfit(tc, y, deg)

    # evaluation function
    def eval_best_fit_poly(t_query, coeffs=coeffs, t_mean=t_mean):
        t_query = n.asarray(t_query, dtype=float)
        return n.polyval(coeffs, t_query - t_mean)
    return(eval_best_fit_poly)
## evaluate fit
#y_fit = eval_best_fit_poly(t)
def f2velocity(xfun,yfun,zfun,t,dt=0.1):
    return(n.array([ (xfun(t+dt)-xfun(t))/dt,
                     (yfun(t+dt)-yfun(t))/dt,
                     (zfun(t+dt)-zfun(t))/dt]))
def pos_err_hist():
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()
    pos_err=[]
    for i in range(len(fragment_ids)):
        pos_err=n.concatenate((pos_err,fragment_pos_err[i]))
    plt.hist(pos_err,bins=100)
    plt.title(n.percentile(pos_err,[67,95]))
    plt.show()

def plot_aspect():
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()
    rlat,rlon,ralt,rsnr,rtime,bragg_enu,rdop,txid,rxid=get_radar_detections()
    rthresh=10
    print(fragment_ids)
    # create a model fit for fragment 1
    pos_ecef=jcoord.geodetic2ecef(fragment_geo_pos[0][:,0],fragment_geo_pos[0][:,1],fragment_geo_pos[0][:,2])
    #print(pos_ecef.shape)
    #exit(0)
    xfun=polyfit_pos(fragment_times[0],pos_ecef[0,:])
    yfun=polyfit_pos(fragment_times[0],pos_ecef[1,:])
    zfun=polyfit_pos(fragment_times[0],pos_ecef[2,:])

    tv=n.sort(fragment_times[0])
    if False:
        plt.subplot(131)
        plt.plot(fragment_times[0],pos_ecef[0,:],".")
        
        plt.plot(tv,xfun(tv))
        plt.subplot(132)
        plt.plot(fragment_times[0],pos_ecef[1,:],".")
        plt.plot(tv,yfun(tv))
        plt.subplot(133)
        plt.plot(fragment_times[0],pos_ecef[2,:],".")
        plt.plot(tv,zfun(tv))
        
        plt.show()

    angles=[]
    snrs=[]
    dopplers=[]
    frlat=[]
    frlon=[]
    fralt=[]
    frtime=[]
    for i in range(len(rlat)):
        for j in range(len(rlat[i])):
            pos_ecef=jcoord.geodetic2ecef(rlat[i][j], rlon[i][j], ralt[i][j])
            rpos_ecef=n.array([xfun(rtime[i][j]),yfun(rtime[i][j]),zfun(rtime[i][j])])
            pos_diff=n.linalg.norm(rpos_ecef-pos_ecef)/1e3
            if pos_diff > 30:
                continue
          #  print(pos_diff)
            bragg_ecef=jcoord.enu2ecef(rlat[i][j], rlon[i][j], ralt[i][j], bragg_enu[i][j,0], bragg_enu[i][j,1], bragg_enu[i][j,2])
            #print(bragg_ecef)
            vfrag=f2velocity(xfun,yfun,zfun,rtime[i][j])
            angle=180*n.arccos(n.dot(bragg_ecef,vfrag)/(n.linalg.norm(bragg_ecef)*n.linalg.norm(vfrag)))/n.pi
            angles.append(angle)
            snrs.append(rsnr[i][j])
            dopplers.append(rdop[i][j])
            llh=jcoord.ecef2geodetic(rpos_ecef[0],rpos_ecef[1],rpos_ecef[2])
            frlat.append(rlat[i][j])
            frlon.append(rlon[i][j])
            fralt.append(ralt[i][j])
            frtime.append(rtime[i][j])

            #print(angle)
 #       plt.plot(rlon[i],rlat[i],"x")
  #  plt.plot(fragment_geo_pos[0][:,1],fragment_geo_pos[0][:,0],".")

   # plt.title("Framgent 1")
    #plt.show()
    fralt=n.array(fralt)
    plt.scatter(frlon,fralt/1e3,c=angles,vmin=60,vmax=120,cmap="coolwarm",s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Altitude (km)")
    cb=plt.colorbar()
    cb.set_label("Aspect angle (deg)")
    plt.show()
    plt.scatter(frlon,fralt/1e3,c=dopplers,vmin=-50,vmax=50,cmap="coolwarm",s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Altitude (km)")
    cb=plt.colorbar()
    cb.set_label("Doppler shift (Hz)")
    plt.show()

    plt.subplot(121)
    plt.scatter(frlon,frlat,c=angles,vmin=60,vmax=120,cmap="coolwarm",s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    cb=plt.colorbar()
    cb.set_label("Aspect angle (deg)")
    plt.subplot(122)
    plt.scatter(frlon,frlat,c=snrs,vmin=-40,vmax=40,s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    cb=plt.colorbar()
    cb.set_label("Power (dB)")
    plt.show()
    plt.scatter(angles,snrs,c=dopplers,vmin=-10,vmax=10,cmap="coolwarm",s=4)
    cb=plt.colorbar()
    cb.set_label("Doppler shift (Hz)")
    plt.xlabel("Aspect angle (deg)")
    plt.ylabel("Peak power (dB)")
    plt.title("Radar echo power")
    plt.show()
    import scipy.constants as c
    #lam=c.c/32.55e6
    #df=2*f*v/c
    vdop= n.array(dopplers)*c.c/2/32.55e6
    plt.subplot(121)
    plt.scatter(frlon,frlat,c=vdop,vmin=-100,vmax=100,cmap="coolwarm",s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    cb=plt.colorbar()
    cb.set_label("Doppler velocity (m/s)")
    plt.subplot(122)
    plt.scatter(frlon,frlat,c=snrs,vmin=-40,vmax=40,s=4)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    cb=plt.colorbar()
    cb.set_label("Power (dB)")
    plt.show()
    plt.scatter(angles,snrs,c=dopplers,vmin=-10,vmax=10,cmap="coolwarm",s=4)
    cb=plt.colorbar()
    cb.set_label("Doppler shift (Hz)")
    plt.xlabel("Aspect angle (deg)")
    plt.ylabel("Peak power (dB)")
    plt.title("Radar echo power")
    plt.show()


def plot_threeplot():
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()
    rlat,rlon,ralt,rsnr,rtime,bragg_enu,rdop,txid,rxid=get_radar_detections()
    rthresh=10



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
                "x", alpha=1.0, color="black")

    for i in range(len(fragment_times)):
        ax.plot(unix_to_datetime(fragment_times[i]),
                fragment_geo_pos[i][:, 0], ".",color="C%d"%(i))#, label="%s"%(fragment_ids[i]))
        ax.text(unix_to_datetime([fragment_times[i][0]]),0.1+fragment_geo_pos[i][0,0],fragment_ids[i],color="C%d"%(i))

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
                ".", color="C%d"%(i))
        ax.text(unix_to_datetime([fragment_times[i][0]]),
                fragment_geo_pos[i][0, 1]+1,
                fragment_ids[i], color="C%d"%(i))

    ax.set_ylabel("Longitude (deg)")
    ax.set_title("(b)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8),
                            sharex=True,
                            constrained_layout=True)
    # =========================================================
    # (c) Altitude vs Time
    # =========================================================
    #ax = axs[2]

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
                    ".", color="C%d"%(i), label="Optical")
        else:
            ax.plot(unix_to_datetime(fragment_times[i]),
                    fragment_geo_pos[i][:, 2] / 1e3,
                    ".", color="C%d"%(i))
        mti=n.argmin(fragment_times[i][:])
        ax.text(unix_to_datetime([fragment_times[i][mti]]),1+fragment_geo_pos[i][mti,2]/1e3,fragment_ids[i],color="C%d"%(i))

    ax.set_ylim([30,90])
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(frameon=False)
    ax.set_title("(c)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.show()
def get_fragment_initial_detection_heights_km(fragment_geo_pos, fragment_times):
    initial_heights_km = []
    for geo, times in zip(fragment_geo_pos, fragment_times):
        if len(times) == 0 or len(geo) == 0:
            continue
        idx = int(n.argmin(n.asarray(times, dtype=float)))
        height_km = float(geo[idx, 2] / 1e3)
        if n.isfinite(height_km):
            initial_heights_km.append(height_km)
    return n.asarray(initial_heights_km, dtype=float)


def get_radar_detection_heights_km(ralt,rsnr):
    radar_heights_km = n.concatenate(ralt)/1e3
    radar_snr = n.concatenate(rsnr)
    radar_heights_km=radar_heights_km[n.where(radar_snr>10)[0]]
    return(radar_heights_km)


def plot_height_hists(save_path="fragment_radar_height_hist.pdf", show=False):
    _, _, _, _, _, fragment_geo_pos, fragment_times = get_fragments()
    #_, _, ralt, _, _, _, _, _, _ = get_radar_detections()
    rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop,txid,rxid = get_radar_detections()
    plt.hist(10.0*n.log10(n.concatenate(rsnr)),bins=50)
    plt.show()


    fragment_initial_heights_km = get_fragment_initial_detection_heights_km(
        fragment_geo_pos,
        fragment_times,
    )
    radar_heights_km = get_radar_detection_heights_km(ralt,rsnr)
    all_heights = n.concatenate((fragment_initial_heights_km, radar_heights_km))
    all_heights = all_heights[n.isfinite(all_heights)]
    if all_heights.size == 0:
        raise ValueError("No valid optical or radar heights available for histogram plot.")

    bin_width_km = 2.0
    hmin = bin_width_km * n.floor(n.min(all_heights) / bin_width_km)
    hmax = bin_width_km * n.ceil(n.max(all_heights) / bin_width_km)
    bins = n.arange(hmin, hmax + bin_width_km, bin_width_km)
    if bins.size < 2:
        bins = n.array([hmin, hmin + bin_width_km], dtype=float)

    optical_color = "#6b6b6b"
    radar_color = "#cb181d"

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
        }
    ):
        fig, ax1 = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)

        ax1.hist(
            fragment_initial_heights_km,
            bins=bins,
            orientation="horizontal",
            color=optical_color,
            alpha=0.55,
            edgecolor=optical_color,
            linewidth=0.8,
        )
        ax1.set_xlabel("Number of optical detections")
        ax1.set_ylabel("Altitude (km)")
        ax1.tick_params(axis="x", colors=optical_color)
        ax1.spines["bottom"].set_color(optical_color)
        ax1.spines["top"].set_visible(False)
        ax1.grid(axis="y", color="0.88", linewidth=0.8)

        ax2 = ax1.twiny()
        ax2.hist(
            radar_heights_km,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color=radar_color,
            linewidth=1.8,
        )
        ax2.set_xlabel("Number of radar detections")
        ax2.tick_params(axis="x", colors=radar_color)
        ax2.spines["top"].set_color(radar_color)
        ax2.spines["bottom"].set_visible(False)

        ax1.set_ylim(hmin, hmax)

        legend_handles = [
            Patch(
                facecolor=optical_color,
                edgecolor=optical_color,
                alpha=0.55,
                label="Initial altitude of optical detection",
            ),
            Line2D(
                [0],
                [0],
                color=radar_color,
                linewidth=1.8,
                label="Radar detection",
            ),
        ]
        ax1.legend(handles=legend_handles, frameon=False, loc="upper right")

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

frags={
    "O1": {
        "place": "Komorniki, PL",
        "lat": 52.3386,
        "lon": 16.8106,
        "type": "wrapped vessel"
    },
    "O2": {
        "place": "Wiry, PL",
        "lat": 52.3072,
        "lon": 16.8574,
        "type": "wrapped vessel"
    },
    "O3": {
        "place": "Śliwno, PL",
        "lat": 52.4456,
        "lon": 16.5619,
        "type": "wrapped vessel (small)"
    },
    "O4": {
        "place": "Sędziny, PL",
        "lat": 52.3833,
        "lon": 16.5750,
        "type": "wrapped vessel"
    },
    "O5": {
        "place": "Komorniki, PL",
        "lat": 52.3369,
        "lon": 16.8094,
        "type": "fragment of plating"
    },
    "O6": {
        "place": "Sędzinko, PL",
        "lat": 52.4256,
        "lon": 16.6578,
        "type": "fragment of plating"
    },
    "O7": {
        "place": "Łowyń, PL",
        "lat": 52.5983,
        "lon": 16.0606,
        "type": "wrapped vessel"
    },
    "O8": {
        "place": "Krzyżkówko, PL",
        "lat": 52.5717,
        "lon": 16.1194,
        "type": "wrapped vessel"
    },
    "O9": {
        "place": "Gołuski, PL",
        "lat": 52.3636,
        "lon": 16.6925,
        "type": "fragment of plating"
    }
}
def plot_map():
        import scipy.io as sio
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # load camera locations from .mat files (if present)
        fl = glob.glob("*.mat")
        cam_lats = []
        cam_lons = []
        for f in fl:
                cal = sio.loadmat(f)
                if "long_lat" in cal:
                        long = cal["long_lat"][0, 0]
                        lat = cal["long_lat"][0, 1]
                        cam_lats.append(lat)
                        cam_lons.append(long)
        import simone_conf
        station_coords=simone_conf.station_coords
       # station_coords = {
       #         "tx": {
       #                 "jruh": [54.63042836, 13.37402466],
       #                 "kborn": [54.118309, 11.769558],
       #         },
       #         "rx": {
       #                 "bornholm": [55.094581, 14.741921],
       #                 "bornim": [52.438, 13.017],
       #                 "hagenow": [53.389, 11.229],
       #                 "moitin": [53.9833666, 11.7248818],
       #                 "neustrelitz": [53.379, 13.067],},
        #}

        # load data
        hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = get_fragments()
        rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop,txid,rxid = get_radar_detections()
        rthresh = 10

        # Single-column journal figure: width ~3.5 in, tall to occupy page space
        fig = plt.figure(figsize=(7, 4.0), constrained_layout=True, dpi=300)
        proj = ccrs.LambertConformal(central_longitude=8.0, central_latitude=53.5, standard_parallels=(50, 56))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([-5, 23, 49.5, 58], crs=ccrs.PlateCarree())

        # high-resolution coastlines/land for publication quality
        land = cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="none", facecolor="lightgray")
        ocean = cfeature.NaturalEarthFeature("physical", "ocean", "10m", edgecolor="none", facecolor="whitesmoke")
        ax.add_feature(land, zorder=0)
        ax.add_feature(ocean, zorder=0)
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":", linewidth=0.4)
        ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.6)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.4)

        # Radar detections (only above threshold)
        shown_radar_label = False
        for i in range(len(rtime)):
                ridx = n.where(rsnr[i] > rthresh)[0]
                if ridx.size == 0:
                        continue
                label = "Radar detection" if not shown_radar_label else ""
                ax.plot(rlon[i][ridx], rlat[i][ridx], "x", color="black", markersize=6,
                                transform=ccrs.PlateCarree(), label=label, zorder=12, alpha=0.9)
                shown_radar_label = True

        # Optical fragment tracks
        shown_optical_label = False
        for fp in range(len(fragment_pos)):
                lat = fragment_geo_pos[fp][:, 0]
                lon = fragment_geo_pos[fp][:, 1]
                label = "Optical detection" if not shown_optical_label else ""
                ax.plot(lon, lat, ".", transform=ccrs.PlateCarree(), ms=3, color="C1", label=label, zorder=11)
                shown_optical_label = True

        # Transmitters (Tx) as red triangles with label
        shown_tx = False
        for name, coords in station_coords.get("tx", {}).items():
                lat, lon = coords[0], coords[1]
                label = "Tx" if not shown_tx else ""
                ax.plot(lon, lat, "^", color="red", markersize=6, transform=ccrs.PlateCarree(), label=label, zorder=14)
                shown_tx = True

        # Receivers (Rx) as blue circles
        shown_rx = False
        for name, coords in station_coords.get("rx", {}).items():
                lat, lon = coords[0], coords[1]
                label = "Rx" if not shown_rx else ""
                ax.plot(lon, lat, "o", color="navy", markersize=5, transform=ccrs.PlateCarree(), label=label, zorder=14)
                shown_rx = True

        # Camera locations (green)
        if cam_lats and cam_lons:
                ax.plot(cam_lons, cam_lats, "o", color="green", markersize=5, transform=ccrs.PlateCarree(), label="Camera", zorder=13)

        # Ground recovered fragments (from frags dict) as gold stars, label with O1..O9
        for key, info in frags.items():
                lat = info["lat"]
                lon = info["lon"]
                ax.plot(lon, lat, marker="*", color="gold", markeredgecolor="black",
                                markersize=9, markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=16)
               # ax.text(lon + 0.08, lat + 0.08, key, transform=ccrs.PlateCarree(),
                #                fontsize=6, fontweight="bold", zorder=17)

        predicted_impacts = get_predicted_impacts()
        shown_predicted_label = False
        for lat, lon, _ in predicted_impacts:
                label = "Predicted impact" if not shown_predicted_label else ""
                ax.plot(
                        lon,
                        lat,
                        marker="*",
                        color="red",
                        markeredgecolor="black",
                        markersize=9,
                        markeredgewidth=0.8,
                        linestyle="None",
                        transform=ccrs.PlateCarree(),
                        zorder=16,
                        label=label,
                )
                shown_predicted_label = True

        # Aesthetic labels and legend
        ax.set_title("Observations of Falcon 9 fragments (2025-02-19)", fontsize=10, pad=6)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)

        # Build legend: include recovered fragment marker and citation
        handles, labels = ax.get_legend_handles_labels()
        recovered_handle = Line2D([0], [0], marker='*', color='gold', markeredgecolor='black', markersize=9, linestyle='None', label='Recovered fragment (ground)')
        citation_handle = Line2D([0], [0], linestyle='none', marker='None', label='Kruzynski et.al., 2025')
        handles.extend([recovered_handle, citation_handle])
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, frameon=True, fontsize=7, loc="upper left", handlelength=1.0)

        # Gridlines with labels (publication-friendly)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        # Only show a single set of longitude labels: use gridliner labels and hide Matplotlib x tick labels
        ax.tick_params(axis="x", which="both", labelbottom=False)

        # Save high-resolution vector PDF for journal submission and also show
        outname = "fig_map_falcon9.pdf"
        plt.savefig(outname, dpi=300, bbox_inches="tight", transparent=False)
        print("Saved map to", outname)
#    plt.show()

def plot_energyloss():
    from pymsis import msis
    msis_date0=n.datetime64("2025-02-19T03:30")
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()

    for i in range(len(fragment_ids)):
        geo=fragment_geo_pos[i]
        alt=geo[:,2]/1e3
        # fit a polynomial to geo position vs time
        pos_ecef=jcoord.geodetic2ecef(geo[:,0],geo[:,1],geo[:,2])
        xfun=polyfit_pos(fragment_times[i],pos_ecef[0,:],deg=3)
        yfun=polyfit_pos(fragment_times[i],pos_ecef[1,:],deg=3)
        zfun=polyfit_pos(fragment_times[i],pos_ecef[2,:],deg=3)
        tv=n.sort(fragment_times[i])
        
        data=msis.run(msis_date0, -5, 40.0, alt, geomagnetic_activity=-1)
        rho_a=data[0,0,0,:,0]
        print(rho_a.shape)
        if False:
            plt.plot(tv,xfun(tv))
            plt.plot(fragment_times[i],pos_ecef[0,:],".")
            plt.show()
            plt.plot(tv,yfun(tv))
            plt.plot(fragment_times[i],pos_ecef[1,:],".")
            plt.show()
            plt.plot(tv,zfun(tv))
            plt.plot(fragment_times[i],pos_ecef[2,:],".")
            plt.show()
        velocity=n.array([f2velocity(xfun,yfun,zfun,t) for t in tv])
        speed=n.linalg.norm(velocity,axis=1)
        El=speed**3 * rho_a 
        plt.scatter(geo[:,1],geo[:,2]/1e3,c=n.log10(El),vmin=5,vmax=9)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    plot_aspect()

    plot_threeplot()

    plot_energyloss()
    exit(0)
    pos_err_hist()


    #plot_map()
    #exit(0)
    plot_threeplot()
    plot_height_hists()

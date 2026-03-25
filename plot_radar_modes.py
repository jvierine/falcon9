import numpy as n
import h5py
import glob 
import re
import jcoord
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timezone
import os
import argparse

# ---- Convert unix seconds to datetime ----
def unix_to_datetime(t):
    return [datetime.utcfromtimestamp(tt) for tt in t]

def parse_time_arg(time_str, ref_date):
    """
    Parses hh:mm :ss or ISO format.
    If hh:mm:ss, combines with ref_date using UTC.
    """
    try:
        # Try hh:mm:ss
        t_obj = datetime.strptime(time_str, "%H:%M:%S").time()
        dt = datetime.combine(ref_date, t_obj)
        # Force to UTC timestamp
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        try:
            # Try ISO
            dt = datetime.fromisoformat(time_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            # Try hh:mm
            try:
                t_obj = datetime.strptime(time_str, "%H:%M").time()
                dt = datetime.combine(ref_date, t_obj)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                raise ValueError(f"Invalid time format: '{time_str}'. Expected hh:mm:ss or ISO format.")

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
        fragment_times.append(n.array(times))
        
    hgt_count=n.zeros(120)
    hgt_count_all=n.zeros(120)

    for fp in range(len(fragment_pos)):
        geo=fragment_geo_pos[fp]
        hgt_count[:]=0
        if len(geo) > 0:
            idxs = n.array(n.round(geo[:,2]/1e3),dtype=int)
            idxs = n.clip(idxs, 0, 119)
            hgt_count[idxs]=1
        hgt_count_all+=hgt_count
    return(hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times)

def get_radar_detections(directory="SIMONe_geodetic_Falcon_2025"):
    fl=glob.glob(os.path.join(directory, "*.h5"))
    fl.sort()
    alts=[]
    lats=[]
    lons=[]
    snrs=[]
    times=[]
    modes=[]
    
    data_date = None
    
    for f in fl:
        try:
            h=h5py.File(f,"r")
            alts.append(h["altitude_m"][()])
            lats.append(h["latitude"][()])
            lons.append(h["longitude"][()])
            snrs.append(h["peak_power_db"][()])
            ts = h["time_unix"][()]
            times.append(ts)
            h.close()
            
            if data_date is None and len(ts) > 0:
                # Use UTC for date detection
                data_date = datetime.fromtimestamp(ts[0], tz=timezone.utc).date()
            
            basename = os.path.basename(f).lower()
            if "miso" in basename:
                modes.append("MISO")
            elif "mimo" in basename:
                modes.append("MIMO")
            elif "simo" in basename:
                modes.append("SIMO")
            else:
                modes.append("Unknown")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return(lats,lons,alts,snrs,times,modes,data_date)


"""
Examples
python plot_radar_modes.py --time-lim 03:45:50 03:47:00 --time-view 03:45:50 03:47:00 --alt-lim 40 75 --lon-view 9 15 --lat-view 52.5 53.25 --lat-lim 52.5 53.25 --snr -20 --time-offset 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot radar fragments with mode-specific coloring.")
    parser.add_argument("--snr", type=float, default=-20.0, help="SNR threshold for filtering radar detections (default: -20.0)")
    parser.add_argument("--dir", type=str, default="figures_radar", help="Directory to save figures (default: figures_radar)")
    parser.add_argument("--time-offset", type=float, default=0.0, help="Time offset in seconds to add to radar data (default: 0.0)")
    
    # Filter arguments
    parser.add_argument("--lat-lim", type=float, nargs=2, help="Latitude filtering limits (min max)")
    parser.add_argument("--lon-lim", type=float, nargs=2, help="Longitude filtering limits (min max)")
    parser.add_argument("--alt-lim", type=float, nargs=2, help="Altitude filtering limits in km (min max)")
    parser.add_argument("--time-lim", type=str, nargs=2, help="Time filtering limits in hh:mm:ss or ISO format")
    
    # Plotting view arguments
    parser.add_argument("--lat-view", type=float, nargs=2, help="Latitude plotting limits (min max)")
    parser.add_argument("--lon-view", type=float, nargs=2, help="Longitude plotting limits (min max)")
    parser.add_argument("--alt-view", type=float, nargs=2, help="Altitude plotting limits in km (min max)")
    parser.add_argument("--time-view", type=str, nargs=2, help="Time plotting limits in hh:mm:ss or ISO format")
    
    args = parser.parse_args()

    snrthresh = args.snr
    save_dir = args.dir
    time_offset = args.time_offset
    fname_suffix = f"_dt={time_offset}s"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Gathering data with SNR threshold = {snrthresh} and time offset = {time_offset}s...")
    hgt_count,hgt_count_all,fragment_ids,fragment_pos,fragment_pos_err,fragment_geo_pos,fragment_times=get_fragments()
    rlat,rlon,ralt,rsnr,rtime,rmode, data_date = get_radar_detections()

    if data_date is None:
        data_date = date.today() # Fallback

    # Apply time offset to radar data
    if time_offset != 0.0:
        for i in range(len(rtime)):
            rtime[i] = rtime[i] + time_offset

    # Combine fragment data for fitting
    all_f_t = []
    all_f_lat = []
    all_f_lon = []
    all_f_alt = []
    for fp in range(len(fragment_times)):
        all_f_t.extend(fragment_times[fp])
        all_f_lat.extend(fragment_geo_pos[fp][:,0])
        all_f_lon.extend(fragment_geo_pos[fp][:,1])
        all_f_alt.extend(fragment_geo_pos[fp][:,2])
    
    all_f_t = n.array(all_f_t)
    all_f_lat = n.array(all_f_lat)
    all_f_lon = n.array(all_f_lon)
    all_f_alt = n.array(all_f_alt)
    
    # Sort by time
    sidx = n.argsort(all_f_t)
    all_f_t = all_f_t[sidx]
    all_f_lat = all_f_lat[sidx]
    all_f_lon = all_f_lon[sidx]
    all_f_alt = all_f_alt[sidx]
    
    # Center time for numerical stability
    t0 = n.mean(all_f_t)
    dt_f = all_f_t - t0

    # Fit parabolic curve (2nd degree polynomial)
    p_lat = n.polyfit(dt_f, all_f_lat, 2)
    p_lon = n.polyfit(dt_f, all_f_lon, 2)
    p_alt = n.polyfit(dt_f, all_f_alt, 2)

    def fit_lat(t): return n.polyval(p_lat, t - t0)
    def fit_lon(t): return n.polyval(p_lon, t - t0)
    def fit_alt(t): return n.polyval(p_alt, t - t0)

    # Apply filters
    def apply_filters(i):
        mask = rsnr[i] > snrthresh
        if args.lat_lim:
            mask &= (rlat[i] >= args.lat_lim[0]) & (rlat[i] <= args.lat_lim[1])
        if args.lon_lim:
            mask &= (rlon[i] >= args.lon_lim[0]) & (rlon[i] <= args.lon_lim[1])
        if args.alt_lim:
            mask &= (ralt[i]/1e3 >= args.alt_lim[0]) & (ralt[i]/1e3 <= args.alt_lim[1])
        if args.time_lim:
            t_min = parse_time_arg(args.time_lim[0], data_date)
            t_max = parse_time_arg(args.time_lim[1], data_date)
            mask &= (rtime[i] >= t_min) & (rtime[i] <= t_max)
        return n.where(mask)[0]

    mode_colors = {
        "MISO": "C1",
        "SIMO": "C2",
        "MIMO": "C3",
        "Unknown": "gray"
    }

    print("Generating Summary Plot (a, b, c) with Parabolic Fit...")
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 8.0), 
                            sharex=True,
                            constrained_layout=True)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)

    # Time range for fit plotting
    t_plot = n.linspace(all_f_t[0], all_f_t[-1], 200)

    # (a) Latitude vs Time
    ax = axs[0]
    # Plots fragments first
    for i in range(len(fragment_times)):
        ax.plot(unix_to_datetime(fragment_times[i]),
                fragment_geo_pos[i][:, 0], "o", color="C0", ms=2, alpha=0.3)
    # Plots radar on top
    for i in range(len(rtime)):
        ridx = apply_filters(i)
        if len(ridx) > 0:
            ax.plot(unix_to_datetime(rtime[i][ridx]),
                    rlat[i][ridx],
                    "o", alpha=0.2, color=mode_colors.get(rmode[i], "black"), ms=2)
    ax.plot(unix_to_datetime(t_plot), fit_lat(t_plot), "r--", linewidth=1.0, label="Fit")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"(a) $\Delta t = {time_offset}$s")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if args.lat_view: ax.set_ylim(args.lat_view)

    # (b) Longitude vs Time
    ax = axs[1]
    # Plots fragments first
    for i in range(len(fragment_times)):
        ax.plot(unix_to_datetime(fragment_times[i]),
                fragment_geo_pos[i][:, 1], "o", color="C0", ms=2, alpha=0.3)
    # Plots radar on top
    for i in range(len(rtime)):
        ridx = apply_filters(i)
        if len(ridx) > 0:
            ax.plot(unix_to_datetime(rtime[i][ridx]),
                    rlon[i][ridx],
                    "o", alpha=0.2, color=mode_colors.get(rmode[i], "black"), ms=2)
    ax.plot(unix_to_datetime(t_plot), fit_lon(t_plot), "r--", linewidth=1.0)
    ax.set_ylabel("Longitude (deg)")
    ax.set_title("(b)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if args.lon_view: ax.set_ylim(args.lon_view)

    # (c) Altitude vs Time
    ax = axs[2]
    plotted_modes = set()
    # Plots fragments first
    for i in range(len(fragment_times)):
        if i == 0:
            ax.plot(unix_to_datetime(fragment_times[i]),
                    fragment_geo_pos[i][:, 2] / 1e3,
                    "o", color="C0", label="Fragment", ms=2, alpha=0.3)
        else:
            ax.plot(unix_to_datetime(fragment_times[i]),
                    fragment_geo_pos[i][:, 2] / 1e3,
                    "o", color="C0", ms=2, alpha=0.3)
    # Plots radar on top
    for i in range(len(rtime)):
        ridx = apply_filters(i)
        if len(ridx) > 0:
            label = None
            if rmode[i] not in plotted_modes:
                label = rmode[i]
                plotted_modes.add(rmode[i])
            ax.plot(unix_to_datetime(rtime[i][ridx]),
                    ralt[i][ridx] / 1e3,
                    "o", alpha=0.2, color=mode_colors.get(rmode[i], "black"), 
                    label=label, ms=2)
    ax.plot(unix_to_datetime(t_plot), fit_alt(t_plot)/1e3, "r--", linewidth=1.0)
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("(c)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if args.alt_view: ax.set_ylim(args.alt_view)
    if args.time_view:
        t_view_min = datetime.fromtimestamp(parse_time_arg(args.time_view[0], data_date), tz=timezone.utc)
        t_view_max = datetime.fromtimestamp(parse_time_arg(args.time_view[1], data_date), tz=timezone.utc)
        ax.set_xlim(t_view_min, t_view_max)

    summary_path = os.path.join(save_dir, f"radar_modes_summary{fname_suffix}.png")
    plt.savefig(summary_path)
    print(f"Saved {summary_path}")

    # =========================================================
    # Deviation Plot
    # =========================================================
    print("Generating Deviation Plot...")
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 8.0), sharex=True, constrained_layout=True)
    
    # Use mean latitude for lon -> km conversion
    mean_lat = n.mean(all_f_lat)
    deg_to_km_lat = 111.32
    deg_to_km_lon = 111.32 * n.cos(n.radians(mean_lat))

    # First plot fragment deviations as background
    axs[0].plot(unix_to_datetime(all_f_t), (all_f_lat - fit_lat(all_f_t)) * deg_to_km_lat, "o", color="C0", ms=2, alpha=0.3, label="Fragment")
    axs[1].plot(unix_to_datetime(all_f_t), (all_f_lon - fit_lon(all_f_t)) * deg_to_km_lon, "o", color="C0", ms=2, alpha=0.3)
    axs[2].plot(unix_to_datetime(all_f_t), (all_f_alt - fit_alt(all_f_t))/1e3, "o", color="C0", ms=2, alpha=0.3)

    plotted_modes = set()
    for i in range(len(rtime)):
        ridx = apply_filters(i)
        if len(ridx) > 0:
            label = None
            if rmode[i] not in plotted_modes:
                label = rmode[i]
                plotted_modes.add(rmode[i])
            
            t_radar = rtime[i][ridx]
            lat_ref = fit_lat(t_radar)
            lon_ref = fit_lon(t_radar)
            alt_ref = fit_alt(t_radar)
            
            axs[0].plot(unix_to_datetime(t_radar), (rlat[i][ridx] - lat_ref) * deg_to_km_lat, 
                        "o", ms=2, alpha=0.3, color=mode_colors.get(rmode[i], "black"), label=label)
            axs[1].plot(unix_to_datetime(t_radar), (rlon[i][ridx] - lon_ref) * deg_to_km_lon, 
                        "o", ms=2, alpha=0.3, color=mode_colors.get(rmode[i], "black"))
            axs[2].plot(unix_to_datetime(t_radar), (ralt[i][ridx] - alt_ref)/1e3, 
                        "o", ms=2, alpha=0.3, color=mode_colors.get(rmode[i], "black"))

    axs[0].set_ylabel("Lat Dev (km)")
    axs[0].set_title(f"Residuals (Data - Fit), $\Delta t = {time_offset}$s")
    axs[0].set_ylim(-50,50) 

    axs[1].set_ylabel("Lon Dev (km)")
    axs[1].set_ylim(-100,100) 

    axs[2].set_ylabel("Alt Dev (km)")
    axs[2].set_xlabel("Time (UTC)")
    axs[2].set_ylim(-20,20) 

    axs[0].legend(frameon=False, loc="best", ncol=2)
    
    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    if args.time_view:
        axs[2].set_xlim(t_view_min, t_view_max)

    dev_path = os.path.join(save_dir, f"radar_deviations{fname_suffix}.png")
    plt.savefig(dev_path)
    print(f"Saved {dev_path}")


    print("Generating Histogram Plot...")
    mode_alts = {m: [] for m in mode_colors.keys()}
    for i in range(len(ralt)):
        ridx = apply_filters(i)
        if len(ridx) > 0:
            mode_alts[rmode[i]].extend(ralt[i][ridx]/1e3)

    fig, ax1 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    line1, = ax1.plot(hgt_count_all, n.arange(120), "s", ms=4, color="C0", label="Camera")
    ax1.set_xlabel("Number of fragments (camera)")
    ax1.set_ylabel("Height (km)")

    ax2 = ax1.twiny()
    lines = [line1]
    for mode, alts in mode_alts.items():
        if len(alts) > 0:
            counts, bins = n.histogram(alts, bins=n.arange(0, 121, 5))
            height = 0.5 * (bins[:-1] + bins[1:])
            l, = ax2.plot(counts, height, marker="o", color=mode_colors.get(mode, "black"),
                        linestyle="none", ms=3, label=f"Radar ({mode})")
            lines.append(l)
    ax2.set_xlabel("Number of fragments (radar)")
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, loc="best", fontsize=7)
    if args.alt_view: ax1.set_ylim(args.alt_view)

    hist_path = os.path.join(save_dir, f"radar_modes_histogram{fname_suffix}.png")
    plt.savefig(hist_path)
    print(f"Saved {hist_path}")

    print("Generating Map Plot...")
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # Determine map extent
        extents = [-20, 40, 30, 70] # Default
        if args.lon_view and args.lat_view:
            extents = [args.lon_view[0], args.lon_view[1], args.lat_view[0], args.lat_view[1]]
        elif args.lon_view:
            extents[0:2] = args.lon_view
        elif args.lat_view:
            extents[2:4] = args.lat_view
        
        lon_range = extents[1] - extents[0]
        lat_range = extents[3] - extents[2]
        base_width = 8.0
        mean_lat_map = (extents[2] + extents[3]) / 2.0
        aspect = (lat_range / lon_range) / n.cos(n.radians(mean_lat_map))
        fig_height = n.clip(base_width * aspect, 2, 8)
        
        fig = plt.figure(figsize=(base_width, fig_height), constrained_layout=True)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extents, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':') 
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        # Plots fragments first
        for fp in range(len(fragment_pos)):
            if fp==0:
                plt.plot(fragment_geo_pos[fp][:,1], fragment_geo_pos[fp][:,0], "o",
                         color="C0", label="Fragment", ms=2, alpha=0.3)
            else:
                plt.plot(fragment_geo_pos[fp][:,1], fragment_geo_pos[fp][:,0], "o",
                         color="C0", ms=2, alpha=0.3)
        
        # Plots radar positions second (on top)
        plotted_modes = set()
        for i in range(len(rtime)):
            ridx = apply_filters(i)
            if len(ridx) == 0: continue
            label = None
            if rmode[i] not in plotted_modes:
                label = rmode[i]
                plotted_modes.add(rmode[i])
            plt.plot(rlon[i][ridx], rlat[i][ridx], "o", alpha=0.2,
                     color=mode_colors.get(rmode[i], "black"), label=label, ms=2)

        # Overplot fit on map
        plt.plot(fit_lon(t_plot), fit_lat(t_plot), "r--", linewidth=1.0, alpha=0.8)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Fragment tracks and Fit, $\Delta t = {time_offset}$s")
        plt.legend(frameon=False, loc="best")        
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        map_path = os.path.join(save_dir, f"radar_modes_map{fname_suffix}.png")
        plt.savefig(map_path)
        print(f"Saved {map_path}")
    except Exception as e:
        print(f"Cartopy plotting failed: {e}")

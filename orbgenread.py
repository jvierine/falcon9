import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


class OrbGenParser:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.t0 = None
        self.t = None
        self.interp_funcs = {}
        self._load()

    def _extract_column_names(self):
        with open(self.filepath) as f:
            for line in f:
                if "yy-mm-ddThh:mm:ss.sss" in line:
                    cols = line.strip("# \n").split()
                    cols[0] = "time"
                    return cols
        raise RuntimeError("Column header line not found")

    def _load(self):

        colnames = self._extract_column_names()

        df = pd.read_csv(
            self.filepath,
            delim_whitespace=True,
            comment="#",
            header=None,
            names=colnames
        )
        df = df.drop_duplicates(subset="time")

        df["time"] = pd.to_datetime(df["time"])

        # Convert Fortran D notation
        for c in df.columns[1:]:
            df[c] = df[c].astype(str).str.replace("D", "E").astype(float)

        self.df = df

        self.t0 = df["time"].iloc[0]
        self.t = (df["time"] - self.t0).dt.total_seconds().values

        # Build interpolation functions
        for c in df.columns[1:]:
            self.interp_funcs[c] = interp1d(
                self.t,
                df[c].values,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate"
            )

    # -------------------------------------------------
    # Interpolation
    # -------------------------------------------------

    def interpolate_time(self, time_query):
        tq = (pd.to_datetime(time_query) - self.t0).total_seconds()
        return {c: float(f(tq)) for c, f in self.interp_funcs.items()}

    def interpolate_seconds(self, seconds):
        return {c: float(f(seconds)) for c, f in self.interp_funcs.items()}

    def interpolate_vector(self, times):

        result = {"time": self.t0 + pd.to_timedelta(times, unit="s")}

        for c, f in self.interp_funcs.items():
            result[c] = f(times)

        return pd.DataFrame(result)

    # -------------------------------------------------
    # Plotting utilities
    # -------------------------------------------------

    def plot_variable(self, column):
        if column not in self.df.columns:
            raise ValueError(f"{column} not found")

        plt.figure()
        plt.plot(self.t / 60, self.df[column])
        plt.xlabel("Time (minutes)")
        plt.ylabel(column)
        plt.title(column + " vs Time")
        plt.grid(True)
        plt.show()

    def plot_altitude(self):

        if "GAlt(km)" not in self.df.columns:
            raise ValueError("GAlt(km) column not found")

        plt.figure()
        plt.plot(self.t / 60, self.df["GAlt(km)"])
        plt.xlabel("Time (minutes)")
        plt.ylabel("Altitude (km)")
        plt.title("Altitude vs Time")
        plt.grid(True)
        plt.show()

    def plot_heat_flux(self):

        if "qdot(W/m2)" not in self.df.columns:
            raise ValueError("qdot(W/m2) column not found")

        plt.figure()
        plt.plot(self.t / 60, self.df["qdot(W/m2)"])
        plt.xlabel("Time (minutes)")
        plt.ylabel("Heat Flux (W/m²)")
        plt.title("Heat Flux vs Time")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------
    # Reentry interface detection
    # -------------------------------------------------

    def find_reentry_interface(self, altitude_km=120):

        alt = self.df["GAlt(km)"].values

        crossing_indices = np.where(
            (alt[:-1] > altitude_km) & (alt[1:] <= altitude_km)
        )[0]

        if len(crossing_indices) == 0:
            return None

        i = crossing_indices[0]

        # linear interpolation for precise crossing time
        t1, t2 = self.t[i], self.t[i + 1]
        h1, h2 = alt[i], alt[i + 1]

        frac = (altitude_km - h1) / (h2 - h1)
        t_cross = t1 + frac * (t2 - t1)

        return {
            "time": self.t0 + pd.to_timedelta(t_cross, unit="s"),
            "seconds_since_start": t_cross,
            "altitude": altitude_km
        }


if __name__ == "__main__":
    import os
    import plot_fragments as plf
    from matplotlib.lines import Line2D
    hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()

    orb = OrbGenParser("data/orbgen#12-cut.dat")

    start_time = "2025-02-19T03:42:45"
    stop_time = "2025-02-19T03:50:00"

    # Use the dataframe directly
    df = orb.df

    lat = df["GLat(d)"]
    lon = df["GLon(d)"]
    alt = df["GAlt(km)"]
    time = df["time"]

    start = pd.to_datetime(start_time)
    stop = pd.to_datetime(stop_time)

    # Convert to seconds since start of dataset
    t_start = (start - orb.t0).total_seconds()
    t_stop = (stop - orb.t0).total_seconds()

    # Interpolate at 1 second resolution
    times = np.arange(t_start, t_stop + 1, 1)

    # Build interpolated trajectory at 1 s resolution
    interp_df = orb.interpolate_vector(times)

    # Publication-quality rc settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    out_dir = "."
    fn_lon_alt = os.path.join(out_dir, "orbgen_lon_vs_alt.pdf")
    fn_lat_lon = os.path.join(out_dir, "orbgen_lat_vs_lon.pdf")
    fn_alt_time = os.path.join(out_dir, "orbgen_altitude_vs_time.pdf")

    # helper: meters -> degrees conversion for latitude and longitude
    meters_to_deg_lat = 1.0 / 111320.0  # approx
    # Plot 1: lon vs alt (ground track)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    #ax1.plot(lon, alt, "x", label="ESA OrbGen (raw)", zorder=10)
    #ax1.plot(interp_df["GLon(d)"], interp_df["GAlt(km)"], "-", label="ESA OrbGen (interp)", zorder=5)
    custom_offsets={"1":[2,0],
                    "2":[-2,0],
                    "3":[4,0],
                    "4":[-2.5,0],
                    "5":[3,0],
                    "8":[-3,0],
                    "7":[-6,0],
                    "9":[3,0],
                    "c":[-4,-0.2],
                    "a":[-5,0],
                    "h":[-6,0],
                    "g":[-7,0],
                    "o":[3,0],
                    "n":[5,0],
                    "p":[4,0.2],
                    "r":[8,0.4],
                    "s":[4,0.0],
                    "w":[6,0.0],
                    "v":[7,0.3],
                    "x":[2,0.0],
                    "z":[2,0.2],
                    "t":[-6,-0.2],
                    "u":[-6,-0.2],
                    "e":[-7,-0.2],
                    "d":[-7,-0.2],
                    "i":[-6,-0.2],
                    "m":[-6,-0.2],
                    "l":[-8,-0.2],
                    "j":[-10,-0.2],
                    "k":[-12,-0.2],
          }
    for i in range(len(fragment_ids)):
        lon_pts = fragment_geo_pos[i][:, 1]
        lat_pts = fragment_geo_pos[i][:, 0]
        alt_pts_km = fragment_geo_pos[i][:, 2] / 1e3

        # 2-sigma error in meters
        err_m = 2.0 * fragment_pos_err[i]

        # convert to degrees for lon/lat (lon conversion depends on latitude)
        lat_err_deg = err_m * meters_to_deg_lat
        lon_err_deg = err_m * (meters_to_deg_lat / np.cos(np.deg2rad(lat_pts)))

        ax1.errorbar(
            lon_pts,
            alt_pts_km,
            xerr=lon_err_deg,
            yerr=(err_m / 1e3),
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=15,
            ecolor="gray",
            elinewidth=0.8,
            capsize=2,
            color="C%d"%(i)
        )
        fid = fragment_ids[i]
        loni0 = np.argmin(lon_pts)
        alt0 = alt_pts_km[loni0]
        lon0 = lon_pts[loni0]
        # draw a small label with a filled white circle background for better visibility.
        marker_color = "C%d" % (i)
        # draw a white disc behind the label, then the colored text on top
        if fid in custom_offsets.keys():
            offset=custom_offsets[fid]
        else:            
            offset=[2,0]
        ax1.scatter(lon0+offset[1], alt0+offset[0], s=120, color="white", zorder=19)
        ax1.text(
            lon0+offset[1],
            alt0+offset[0],
            fid,
            color=marker_color,
            fontsize=8,
            va="center",
            ha="center",
            zorder=20
        )
        

    # Add vertical lines for recovered ground fragment longitudes
    for idx, (fid, info) in enumerate(frags.items()):
        ax1.axvline(
            info["lon"],
            color="red",
            linestyle="--",
            linewidth=0.8,
            zorder=8,
            label=None  # avoid multiple legend entries
        )
        # small label near the top of the plot for identification
        ylim = ax1.get_ylim()
        y_text = ylim[1] - 0.02 * (ylim[1] - ylim[0])
#        ax1.text(
 #           info["lon"],
  #          y_text,
   #         fid,
    #        color="red",
     #       fontsize=8,
      #      va="top",
       #     ha="center",
        #    rotation=90,
         #   zorder=9
       # )

    # create a single legend entry for the red dashed vertical lines
    handles, labels = ax1.get_legend_handles_labels()
    proxy_line = Line2D([0], [0], color="red", linestyle="--", linewidth=0.8)
    handles.append(proxy_line)
    labels.append("Ground recovered fragment")
    ax1.legend(handles=handles, labels=labels, frameon=False)

    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Ground Track (Longitude vs Altitude)")
    ax1.grid(True, linestyle="--", linewidth=0.5)
    fig1.tight_layout()
    fig1.savefig(fn_lon_alt, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: lat vs lon (ground track)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    #ax2.plot(lon, lat, "x", label="ESA OrbGen (raw)", zorder=10)
    #ax2.plot(interp_df["GLon(d)"], interp_df["GLat(d)"], "-", label="ESA OrbGen (interp)", zorder=5)
    for i in range(len(fragment_ids)):
        lon_pts = fragment_geo_pos[i][:, 1]
        lat_pts = fragment_geo_pos[i][:, 0]

        err_m = 2.0 * fragment_pos_err[i]
        lat_err_deg = err_m * meters_to_deg_lat
        lon_err_deg = err_m * (meters_to_deg_lat / np.cos(np.deg2rad(lat_pts)))

        ax2.errorbar(
            lon_pts,
            lat_pts,
            xerr=lon_err_deg,
            yerr=lat_err_deg,
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=15,
            ecolor="gray",
            elinewidth=0.8,
            capsize=2
        )
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_title("Ground Track (Latitude vs Longitude)")
    ax2.grid(True, linestyle="--", linewidth=0.5)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(fn_lat_lon, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: altitude vs time
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    #ax3.plot(time, alt, "x", label="ESA OrbGen (raw)", zorder=10)
    #ax3.plot(interp_df["time"], interp_df["GAlt(km)"], "-", label="ESA OrbGen (interp)", zorder=5)
    for i in range(len(fragment_ids)):
        tvals = plf.unix_to_datetime(fragment_times[i])
        alt_pts_km = fragment_geo_pos[i][:, 2] / 1e3

        err_m = 2.0 * fragment_pos_err[i]
        alt_err_km = err_m / 1e3

        ax3.errorbar(
            tvals,
            alt_pts_km,
            yerr=alt_err_km,
            fmt=".",
            zorder=15,
            label="Optical detection" if i == 0 else None,
            ecolor="gray",
            elinewidth=0.8,
            capsize=2
        )
    ax3.set_xlabel("Time (UTC)")
    ax3.set_ylabel("Altitude (km)")
    ax3.set_title("Altitude vs Time")
    ax3.grid(True, linestyle="--", linewidth=0.5)

    # Improve time formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.legend(frameon=False)
    fig3.autofmt_xdate()
    fig3.tight_layout()
    #plt.show()
    fig3.savefig(fn_alt_time, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved separate PDF files: {fn_lon_alt}, {fn_lat_lon}, {fn_alt_time}")
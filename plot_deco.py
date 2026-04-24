import glob
import re
from datetime import timezone
from pathlib import Path

import h5py
import jcoord
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as n
import numpy as np
import plot_fragments as pf
import scipy.constants as sc
import scipy.interpolate as sint
import simone_conf
from matplotlib import ticker


DECODED_SAMPLE_INTERVAL_S = 1000 * 10e-6
RANGE_SAMPLE_INTERVAL_S = 10e-6
DEFAULT_RCS_LINKS = [
    ("jruh", "bornim"),
    ("jruh", "bornholm"),
    ("jruh", "hagenow"),
    ("jruh", "moitin"),
    ("kborn", "bornholm"),
    ("kborn", "hagenow"),
    ("kborn", "moitin"),
    ("kborn", "bornim"),
]
LINK_DISPLAY_NAMES = {
    "jruh": "Juliusruh",
    "kborn": "Kühlungsborn",
    "bornim": "Bornim",
    "bornholm": "Bornholm",
    "hagenow": "Hagenow",
    "moitin": "Moitin",
}
def get_fragment_info(tx="kborn",rx="hagenow"):
    lam=sc.c/32.55e6
    station_coords = simone_conf.station_coords
    hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()

    fragment_vel_ecef=[]
    fragment_aspects=[]
    fragment_dops=[]
    fragment_range=[]
    fragment_dts=[]

    for i in range(len(fragment_ids)):
        geo=fragment_geo_pos[i]
        tx_latlon=station_coords["tx"][tx]
        rx_latlon=station_coords["rx"][rx]
        tx_ecef=jcoord.geodetic2ecef(tx_latlon[0],tx_latlon[1],10)
        rx_ecef=jcoord.geodetic2ecef(rx_latlon[0],rx_latlon[1],10)
        #print(geo[:,2])
        ecefs=jcoord.geodetic2ecef(geo[:,0],geo[:,1],geo[:,2])
        #print(ecefs.shape)
        if (n.max(fragment_times[i])-n.min(fragment_times[i])) > 60:
            deg=5
        else:
            deg=1
        xfun=pf.polyfit_pos(fragment_times[i],ecefs[0,:],deg=deg)
        yfun=pf.polyfit_pos(fragment_times[i],ecefs[1,:],deg=deg)
        zfun=pf.polyfit_pos(fragment_times[i],ecefs[2,:],deg=deg)
        #vel_ecef=pf.f2velocity(xfun,yfun,zfun,t,dt=1)
        vel_ecef=n.array([pf.f2velocity(xfun,yfun,zfun,t) for t in fragment_times[i]])
        fragment_vel_ecef.append(vel_ecef)
        #print(vel_ecef.shape)
        
        if fragment_ids[i]=='1':
            one_idx=i
            #plt.figure()
            #plt.plot(fragment_times[i],xfun(fragment_times[i])-ecefs[0,:],".")
            #plt.show()
            #plt.plot(fragment_times[i],yfun(fragment_times[i])-ecefs[1,:],".")
            #plt.show()
            #plt.plot(fragment_times[i],zfun(fragment_times[i])-ecefs[2,:],".")
            #plt.show()

            #plt.plot(fragment_times[i],n.linalg.norm(vel_ecef,axis=1),".")
            #plt.show()
        #plt.plot(vel_ecef)

            # fit 4th deg polynomial to fragment_times[i], ecefs[j,:]
            
        #print(ecefs.shape)
        frag_rgs=[]

        dop=n.zeros(len(fragment_times[i]))
        aspect=n.zeros(len(fragment_times[i]))

        for j in range(ecefs.shape[1]):
            rng=(n.linalg.norm(ecefs[:,j]-tx_ecef)+n.linalg.norm(ecefs[:,j]-rx_ecef))/1e3
            #print(rng)
            frag_rgs.append(rng)
            k=(ecefs[:,j]-tx_ecef)-(rx_ecef-ecefs[:,j])
            k0=k/n.linalg.norm(k)
            #print(n.linalg.norm(k0))
            k=k0*4*n.pi/lam
            #vel_ecef[:,j]
          #  print(n.linalg.norm(vel_ecef[j,:]))

            dop[j]=-n.dot(k,vel_ecef[j,:])/2/n.pi
            aspect[j]=n.arccos(dop[j]/(n.linalg.norm(k)*n.linalg.norm(vel_ecef[j,:])))
        fragment_aspects.append(aspect)
        fragment_dops.append(dop)
        fragment_range.append(frag_rgs)
        tv=n.array((fragment_times[i])*1e9,dtype="datetime64[ns]")
        fragment_dts.append(tv)

    return(fragment_aspects,fragment_dops,fragment_range,fragment_dts)

def _norm_id(x):
    return re.sub(r'[^0-9a-z]', '', str(x).lower())

def sn_plus_n_over_n_to_rcs(sn_plus_n_over_n,
                            R_tx,
                            R_rx,
                            frequency_hz=32.55e6,
                            P_tx=500,
                            G_tx=1,
                            G_rx=1,
                            B_rx=100.0,
                            T_noise=6000):
    """
    Convert measured (S+N)/N to bistatic radar cross section (RCS).

    Supports scalars or numpy arrays.

    Parameters
    ----------
    sn_plus_n_over_n : float or array
        Measured (S+N)/N in linear units
    R_tx : float or array
        Transmitter-to-target range (m)
    R_rx : float or array
        Target-to-receiver range (m)
    frequency_hz : float
        Radar carrier frequency (Hz)

    Returns
    -------
    sigma : float or array
        Radar cross section (m²)
    """

    sn_plus_n_over_n = n.asarray(sn_plus_n_over_n)
    sn_plus_n_over_n[sn_plus_n_over_n<=1.2]=0.0

    c = sc.c#299792458.0
    wavelength = c / frequency_hz

    # Convert (S+N)/N → S/N
    snr = sn_plus_n_over_n - 1.0

    # Noise power
    noise_power = sc.k * T_noise * B_rx

    # Signal power
    signal_power = snr * noise_power

    # Bistatic radar equation solved for RCS
    sigma = (
        signal_power
        * (4 * n.pi)**3
        * R_tx**2
        * R_rx**2
        / (P_tx * G_tx * G_rx * wavelength**2)
    )

    return sigma


def db_to_linear(x_db):
    return 10**(x_db / 10)


def linear_to_db(x):
    return 10 * n.log10(x)


def get_link_display_name(tx, rx):
    tx_name = LINK_DISPLAY_NAMES.get(tx, str(tx))
    rx_name = LINK_DISPLAY_NAMES.get(rx, str(rx))
    return f"{tx_name}-{rx_name}"


def get_decoded_file_paths(tx="jruh", rx="bornim"):
    pattern = f"simone/decoded_files/mmaria_decoded_{tx}_{rx}_*"
    file_paths = sorted(glob.glob(pattern))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No decoded SIMONe files found for {tx}-{rx}.")
    return file_paths


def load_decoded_power(tx="jruh", rx="bornim"):
    file_paths = get_decoded_file_paths(tx=tx, rx=rx)
    ut_parts = []
    power_parts = []
    rgs_km = None

    for file_path in file_paths:
        with h5py.File(file_path, "r") as handle:
            z = handle["decoded_data/voltage"][()] + handle["decoded_data/residual"][()]
            # average polarization and channel
            power_block = n.sum(n.abs(z) ** 2.0, axis=(0, 1))
            chunk_start_unix = float(handle["decoded_data/chunk_start_time_ns"][()]) / 1e9
            tvec = chunk_start_unix + n.arange(power_block.shape[1], dtype=float) * DECODED_SAMPLE_INTERVAL_S

            ut_parts.append(tvec)
            power_parts.append(power_block)

            if rgs_km is None:
                rgs_km = n.arange(power_block.shape[0], dtype=float) * RANGE_SAMPLE_INTERVAL_S * sc.c / 1e3

    ut_unix = n.concatenate(ut_parts)
    power = n.concatenate(power_parts, axis=1)
    ut_dt64 = n.array(ut_unix * 1e9, dtype="datetime64[ns]")
    return {
        "times_unix": ut_unix,
        "times_datetime64": ut_dt64,
        "power": power,
        "range_km": rgs_km,
    }


def compute_snr_from_power(power):
    power = n.asarray(power, dtype=float)
    noise_floor = n.median(power, axis=0)
    noise_floor = n.where(noise_floor > 0.0, noise_floor, 1.0)
    return power / noise_floor[None, :]


def compute_rcs_grid(tx="jruh", rx="bornim"):
    decoded = load_decoded_power(tx=tx, rx=rx)
    snr = compute_snr_from_power(decoded["power"])
    range_km = n.asarray(decoded["range_km"], dtype=float)

    r_tx = n.broadcast_to((0.5 * range_km[:, None]) * 1e3, snr.shape)
    rcs = sn_plus_n_over_n_to_rcs(
        snr,
        r_tx,
        r_tx,
    )
    rcs = n.where(snr < 1.2, 1e-9, rcs)

    decoded["snr"] = snr
    decoded["sn_plus_n_over_n_db"] = 10.0 * n.log10(n.maximum(snr, 1e-12))
    decoded["rcs_m2"] = rcs
    decoded["rcs_dbsm"] = 10.0 * n.log10(n.maximum(rcs, 1e-12))
    return decoded


def _coerce_datetime64(value):
    if value is None:
        return None
    return n.datetime64(value, "ns")


def _slice_time_window(times_dt64, values, start_time=None, end_time=None):
    times_dt64 = n.asarray(times_dt64)
    mask = n.ones(times_dt64.shape, dtype=bool)
    start_dt64 = _coerce_datetime64(start_time)
    end_dt64 = _coerce_datetime64(end_time)
    if start_dt64 is not None:
        mask &= times_dt64 >= start_dt64
    if end_dt64 is not None:
        mask &= times_dt64 <= end_dt64
    if not n.any(mask):
        raise ValueError("The requested time window does not overlap the decoded data.")
    return times_dt64[mask], values[:, mask]


def publication_rcparams():
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }


def plot_decoded(
    tx="jruh",
    rx="bornim",
    start_time=None,
    end_time=None,
    ymin=100,
    ymax=600,
    ax=None,
    add_colorbar=True,
    show=True,
    output_filename="rcs_single_column.pdf",
    title=None,
    cmap="viridis",
    vmin=0,
    vmax=60,
    field_name="rcs_dbsm",
    colorbar_label="RCS (dBsm)",
    precomputed_grid=None,
):
    decoded = compute_rcs_grid(tx=tx, rx=rx) if precomputed_grid is None else precomputed_grid
    times_plot, rcs_dbsm = _slice_time_window(
        decoded["times_datetime64"],
        decoded[field_name],
        start_time=start_time,
        end_time=end_time,
    )

    created_fig = ax is None
    if created_fig:
        with plt.rc_context(publication_rcparams()):
            fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(
        times_plot,
        decoded["range_km"],
        rcs_dbsm,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        rasterized=True,
    )
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Propagation range (km)")
    ax.set_title(get_link_display_name(tx, rx) if title is None else title, pad=4)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(top=False, right=True)

    if created_fig and add_colorbar:
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(colorbar_label)
        cb.ax.tick_params(direction="in", labelsize=7)
        fig.autofmt_xdate(rotation=30, ha="right")

    if created_fig and output_filename is not None:
        fig.savefig(output_filename, bbox_inches="tight")

    if created_fig and show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return {
        "mesh": pcm,
        "times_datetime64": times_plot,
        "range_km": decoded["range_km"],
        "rcs_dbsm": rcs_dbsm,
        "figure": fig,
        "axes": ax,
    }


def plot_rcs_vs_aspect(
    tx="jruh",
    rx="bornim",
    ymin=100,
    ymax=600,
    output_filename="rcs_vs_aspect_single_column.pdf",
    show=True,
):
    decoded = compute_rcs_grid(tx=tx, rx=rx)
    ridx = n.where((decoded["range_km"] > ymin) & (decoded["range_km"] < ymax))[0]
    if ridx.size == 0:
        raise ValueError("No range bins fall inside the requested ymin/ymax interval.")

    rcs_dbsm = decoded["rcs_dbsm"]
    rcs_aspect_db = n.max(rcs_dbsm[ridx, :], axis=0)

    fragment_aspects, _, _, _ = get_fragment_info(tx=tx, rx=rx)
    _, _, fragment_ids, _, _, _, fragment_times = pf.get_fragments()

    with plt.rc_context(publication_rcparams()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)

        for i, fid in enumerate(fragment_ids):
            if fid not in ("1", "2"):
                continue
            aspect_int = sint.interp1d(
                fragment_times[i],
                fragment_aspects[i],
                bounds_error=False,
                fill_value=n.nan,
            )
            aspect_deg = 180.0 * aspect_int(decoded["times_unix"]) / n.pi
            ax.plot(
                aspect_deg,
                rcs_aspect_db,
                ".",
                markersize=2.5,
                label=fid,
            )

        ax.set_xticks(np.arange(40, 131, 10))
        ax.set_xlabel("Aspect angle (deg)")
        ax.set_ylabel("Peak RCS (dBsm)")
        ax.set_xlim(40, 130)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.tick_params(top=True, right=True, direction="in")
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_title(get_link_display_name(tx, rx), pad=4)
        ax.legend(frameon=False)
        fig.savefig(output_filename, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    plot_decoded(tx="kborn", rx="hagenow")

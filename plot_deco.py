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
import fit_ballistic3 as fb3
import mean_free_path as mfp
import plot_fragments as pf
import scipy.constants as sc
import scipy.interpolate as sint
import simone_conf
from matplotlib import ticker


DECODED_SAMPLE_INTERVAL_S = 1000 * 10e-6
RANGE_SAMPLE_INTERVAL_S = 10e-6
DEFAULT_TIME_SMOOTH_SAMPLES = 20
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
OPTICAL_FRAGMENT_FAMILY_1_IDS = {"1", "3", "5", "9", "n", "o", "p", "r", "s", "w", "v", "x", "z"}
OPTICAL_FRAGMENT_FAMILY_2_IDS = {"2", "4", "7", "8", "a", "c", "h", "g", "i", "m", "j", "l", "k", "t", "u", "e", "d"}


def optical_fragment_family_color(fragment_id):
    fragment_id = str(fragment_id)
    if fragment_id in OPTICAL_FRAGMENT_FAMILY_2_IDS:
        return (1.0, 0.0, 0.0, 0.65)
    return (1.0, 1.0, 1.0, 0.65)


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


def get_optical_fragment_range_points(tx="kborn", rx="hagenow"):
    """Return optical fragment detections as bistatic propagation-range points."""
    station_coords = simone_conf.station_coords
    _, _, fragment_ids, fragment_pos, _, _, fragment_times = pf.get_fragments()

    tx_latlon = station_coords["tx"][tx]
    rx_latlon = station_coords["rx"][rx]
    tx_ecef = n.asarray(jcoord.geodetic2ecef(tx_latlon[0], tx_latlon[1], 10.0), dtype=float)
    rx_ecef = n.asarray(jcoord.geodetic2ecef(rx_latlon[0], rx_latlon[1], 10.0), dtype=float)

    points = []
    for fragment_id, pos_ecef, times_unix in zip(fragment_ids, fragment_pos, fragment_times):
        pos_ecef = n.asarray(pos_ecef, dtype=float)
        times_unix = n.asarray(times_unix, dtype=float)
        if pos_ecef.size == 0 or times_unix.size == 0:
            continue
        ranges_km = (
            n.linalg.norm(pos_ecef - tx_ecef[None, :], axis=1)
            + n.linalg.norm(pos_ecef - rx_ecef[None, :], axis=1)
        ) / 1e3
        times_dt64 = n.asarray(times_unix * 1e9, dtype="datetime64[ns]")
        points.append(
            {
                "fragment_id": fragment_id,
                "times_datetime64": times_dt64,
                "propagation_range_km": ranges_km,
            }
        )
    return points


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


def build_time_smoothing_kernel(time_smooth_samples=None, time_smooth_kernel=None):
    if time_smooth_kernel is not None:
        kernel = n.asarray(time_smooth_kernel, dtype=float).reshape(-1)
    elif time_smooth_samples is not None and int(time_smooth_samples) > 1:
        width = int(time_smooth_samples)
        kernel = n.repeat(1.0 / float(width), width)
    else:
        return None

    if kernel.size < 2:
        return None

    kernel_sum = float(n.sum(kernel))
    if not n.isfinite(kernel_sum) or kernel_sum == 0.0:
        raise ValueError("time smoothing kernel must have a finite, non-zero sum")

    return kernel / kernel_sum


def smooth_time_series_by_range_gate(values, kernel):
    values = n.asarray(values, dtype=float)
    kernel = n.asarray(kernel, dtype=float).reshape(-1)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_range, n_time)")
    if kernel.size < 2:
        return values.copy()

    smoothed = n.empty_like(values)
    for ir in range(values.shape[0]):
        # Centered convolution (`mode="same"`) preserves the time grid.
        smoothed[ir, :] = n.convolve(values[ir, :], kernel, mode="same")
    return smoothed


def compute_rcs_grid(tx="jruh", rx="bornim", time_smooth_samples=None, time_smooth_kernel=None):
    decoded = load_decoded_power(tx=tx, rx=rx)
    snr = compute_snr_from_power(decoded["power"])
    range_km = n.asarray(decoded["range_km"], dtype=float)

    # Keep a copy of raw SN for debugging/inspection
    decoded["snr_raw"] = snr.copy()

    kernel = build_time_smoothing_kernel(
        time_smooth_samples=time_smooth_samples,
        time_smooth_kernel=time_smooth_kernel,
    )
    if kernel is not None:
        snr = smooth_time_series_by_range_gate(snr, kernel)

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


def _select_peak_range_gate(range_km, values, ymin=100, ymax=600):
    range_km = n.asarray(range_km, dtype=float)
    values = n.asarray(values, dtype=float)
    ridx = n.where((range_km >= ymin) & (range_km <= ymax))[0]
    if ridx.size == 0:
        raise ValueError("No range bins fall inside the requested ymin/ymax interval.")

    gate_peak_values = n.nanmax(values[ridx, :], axis=1)
    if not n.any(n.isfinite(gate_peak_values)):
        raise ValueError("No finite values were found in the requested time/range window.")

    best_local_idx = int(n.nanargmax(gate_peak_values))
    gate_idx = int(ridx[best_local_idx])
    return gate_idx, float(range_km[gate_idx]), float(gate_peak_values[best_local_idx])


def fit_exponential_efolding_time(times_dt64, linear_values, fit_start_time=None, fit_end_time=None):
    fit_start_dt64 = _coerce_datetime64(fit_start_time)
    fit_end_dt64 = _coerce_datetime64(fit_end_time)

    times_dt64 = n.asarray(times_dt64)
    linear_values = n.asarray(linear_values, dtype=float)

    mask = n.isfinite(linear_values) & (linear_values > 0.0)
    if fit_start_dt64 is not None:
        mask &= times_dt64 >= fit_start_dt64
    if fit_end_dt64 is not None:
        mask &= times_dt64 <= fit_end_dt64

    if n.count_nonzero(mask) < 3:
        raise ValueError("Need at least three positive samples inside the fit window.")

    fit_times = times_dt64[mask]
    fit_values = linear_values[mask]
    fit_seconds = (
        fit_times.astype("datetime64[ns]").astype("int64")
        - fit_times[0].astype("datetime64[ns]").astype("int64")
    ) * 1e-9
    log_values = n.log(fit_values)

    coeffs, covariance = n.polyfit(fit_seconds, log_values, deg=1, cov=True)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    slope_sigma = float(n.sqrt(max(covariance[0, 0], 0.0)))

    if slope >= 0.0:
        raise ValueError("Fitted exponential slope is non-negative; no decay time can be estimated.")

    tau_s = -1.0 / slope
    tau_sigma_s = abs(slope_sigma / (slope * slope))

    model_values = n.exp(intercept + slope * fit_seconds)
    return {
        "fit_times_datetime64": fit_times,
        "fit_values_linear": fit_values,
        "fit_model_linear": model_values,
        "slope": slope,
        "slope_sigma": slope_sigma,
        "tau_s": tau_s,
        "tau_2sigma_s": 2.0 * tau_sigma_s,
    }


def get_fragment_fit_tx_rx_ranges_km(
    tx,
    rx,
    time_dt64,
    fit_path="ballistic_fit_sharedstart_1.h5",
):
    fit_path = Path(fit_path)
    if not fit_path.is_absolute():
        fit_path = Path(__file__).resolve().parent / fit_path

    with h5py.File(fit_path, "r") as handle:
        group = handle["model"]
        times_model = n.asarray(group["times_model"][()], dtype=float)
        lat_deg = n.asarray(group["lat_deg"][()], dtype=float)
        lon_deg = n.asarray(group["lon_deg"][()], dtype=float)
        hgt_m = n.asarray(group["hgt_m"][()], dtype=float)

    target_times_ns = n.asarray(time_dt64, dtype="datetime64[ns]").astype("int64")
    scalar_input = target_times_ns.ndim == 0
    target_times_ns = n.atleast_1d(target_times_ns)
    target_times_unix = target_times_ns.astype(float) * 1e-9

    tmin = float(n.nanmin(times_model))
    tmax = float(n.nanmax(times_model))
    target_times_unix = n.clip(target_times_unix, tmin, tmax)

    lat_i = n.interp(target_times_unix, times_model, lat_deg)
    lon_i = n.interp(target_times_unix, times_model, lon_deg)
    hgt_i = n.interp(target_times_unix, times_model, hgt_m)

    tx_latlon = simone_conf.station_coords["tx"][tx]
    rx_latlon = simone_conf.station_coords["rx"][rx]
    tx_ecef = n.asarray(jcoord.geodetic2ecef(tx_latlon[0], tx_latlon[1], 10.0), dtype=float)
    rx_ecef = n.asarray(jcoord.geodetic2ecef(rx_latlon[0], rx_latlon[1], 10.0), dtype=float)
    target_ecef = n.asarray(
        [jcoord.geodetic2ecef(lat, lon, hgt) for lat, lon, hgt in zip(lat_i, lon_i, hgt_i)],
        dtype=float,
    )

    r_tx_km = n.linalg.norm(target_ecef - tx_ecef[None, :], axis=1) / 1e3
    r_rx_km = n.linalg.norm(target_ecef - rx_ecef[None, :], axis=1) / 1e3

    result = {
        "time_unix": target_times_unix,
        "lat_deg": lat_i,
        "lon_deg": lon_i,
        "hgt_km": hgt_i / 1e3,
        "r_tx_km": r_tx_km,
        "r_rx_km": r_rx_km,
    }
    if scalar_input:
        return {key: float(value[0]) for key, value in result.items()}
    return result


def load_fragment_fit_model_trajectory(fit_path="ballistic_fit_sharedstart_1.h5"):
    fit_path = Path(fit_path)
    if not fit_path.is_absolute():
        fit_path = Path(__file__).resolve().parent / fit_path

    with h5py.File(fit_path, "r") as handle:
        group = handle["model"]
        return {
            "times_unix": n.asarray(group["times_model"][()], dtype=float),
            "pos_eci": n.asarray(group["pos_eci"][()], dtype=float),
            "vel_eci": n.asarray(group["vel_eci"][()], dtype=float),
            "lat_deg": n.asarray(group["lat_deg"][()], dtype=float),
            "lon_deg": n.asarray(group["lon_deg"][()], dtype=float),
            "hgt_m": n.asarray(group["hgt_m"][()], dtype=float),
        }


def get_fragment_fit_paths(fit_paths=None, pattern="ballistic_fit_sharedstart*.h5"):
    base_dir = Path(__file__).resolve().parent
    if fit_paths is None:
        return sorted(base_dir.glob(pattern))

    if isinstance(fit_paths, (str, Path)):
        fit_paths = [fit_paths]

    resolved = []
    for fit_path in fit_paths:
        fit_path = Path(fit_path)
        if any(char in str(fit_path) for char in "*?[]"):
            search_path = fit_path if fit_path.is_absolute() else base_dir / fit_path
            resolved.extend(sorted(search_path.parent.glob(search_path.name)))
        else:
            resolved.append(fit_path if fit_path.is_absolute() else base_dir / fit_path)
    return resolved


def get_fragment_fit_time_track(fit_path, start_time=None, end_time=None):
    track = load_fragment_fit_model_trajectory(fit_path=fit_path)
    times_unix = n.asarray(track["times_unix"], dtype=float)
    times_dt64 = n.asarray(times_unix * 1e9, dtype="datetime64[ns]")

    mask = n.isfinite(times_unix)
    start_dt64 = _coerce_datetime64(start_time)
    end_dt64 = _coerce_datetime64(end_time)
    if start_dt64 is not None:
        mask &= times_dt64 >= start_dt64
    if end_dt64 is not None:
        mask &= times_dt64 <= end_dt64

    return times_dt64[mask]


def get_fragment_fit_bragg_geometry(
    tx,
    rx,
    time_dt64,
    fit_path="ballistic_fit_sharedstart_1.h5",
):
    track = load_fragment_fit_model_trajectory(fit_path=fit_path)

    target_times_ns = n.asarray(time_dt64, dtype="datetime64[ns]").astype("int64")
    scalar_input = target_times_ns.ndim == 0
    target_times_ns = n.atleast_1d(target_times_ns)
    target_times_unix = target_times_ns.astype(float) * 1e-9

    tmin = float(n.nanmin(track["times_unix"]))
    tmax = float(n.nanmax(track["times_unix"]))
    target_times_unix = n.clip(target_times_unix, tmin, tmax)

    pos_eci = n.column_stack(
        [
            n.interp(target_times_unix, track["times_unix"], track["pos_eci"][:, axis])
            for axis in range(3)
        ]
    )
    vel_eci = n.column_stack(
        [
            n.interp(target_times_unix, track["times_unix"], track["vel_eci"][:, axis])
            for axis in range(3)
        ]
    )

    tx_latlon = simone_conf.station_coords["tx"][tx]
    rx_latlon = simone_conf.station_coords["rx"][rx]
    tx_ecef = n.asarray(jcoord.geodetic2ecef(tx_latlon[0], tx_latlon[1], 10.0), dtype=float)
    rx_ecef = n.asarray(jcoord.geodetic2ecef(rx_latlon[0], rx_latlon[1], 10.0), dtype=float)

    tx_eci = fb3.ecef_to_eci_position(
        n.repeat(tx_ecef[None, :], target_times_unix.size, axis=0),
        target_times_unix,
    )
    rx_eci = fb3.ecef_to_eci_position(
        n.repeat(rx_ecef[None, :], target_times_unix.size, axis=0),
        target_times_unix,
    )

    tx_to_target = pos_eci - tx_eci
    target_to_rx = rx_eci - pos_eci
    tx_to_target_norm = n.linalg.norm(tx_to_target, axis=1)
    target_to_rx_norm = n.linalg.norm(target_to_rx, axis=1)
    u_tx = tx_to_target / n.maximum(tx_to_target_norm[:, None], 1e-12)
    u_rx = target_to_rx / n.maximum(target_to_rx_norm[:, None], 1e-12)
    k_bragg = u_tx - u_rx

    cos_theta = n.sum(k_bragg * vel_eci, axis=1) / (
        n.maximum(n.linalg.norm(k_bragg, axis=1), 1e-12)
        * n.maximum(n.linalg.norm(vel_eci, axis=1), 1e-12)
    )
    cos_theta = n.clip(cos_theta, -1.0, 1.0)
    aspect_deg = n.degrees(n.arccos(cos_theta))

    result = {
        "time_unix": target_times_unix,
        "aspect_deg": aspect_deg,
        "r_tx_km": tx_to_target_norm / 1e3,
        "r_rx_km": target_to_rx_norm / 1e3,
        "propagation_range_km": (tx_to_target_norm + target_to_rx_norm) / 1e3,
    }
    if scalar_input:
        return {key: float(value[0]) for key, value in result.items()}
    return result


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
    fit_path="ballistic_fit_sharedstart_1.h5",
    fit_paths=None,
    overlay_predicted_range=True,
    predicted_range_overlay="ballistic",
    optical_fragment_delay_s=0.0,
    show_aspect_axis=True,
    aspect_time_shift_s=-1.0,
    aspect_tick_values=(130, 110, 90, 70, 50),
    # optional running-mean smoothing in time per range gate
    time_smooth_samples=20,
    time_smooth_kernel=None,
):
    # If caller did not provide a precomputed grid, compute it here.
    # Forward optional smoothing parameters into compute_rcs_grid so smoothing happens
    # on the SN time-series before converting to RCS.
    if precomputed_grid is None:
        decoded = compute_rcs_grid(
            tx=tx,
            rx=rx,
            time_smooth_samples=time_smooth_samples,
            time_smooth_kernel=time_smooth_kernel,
        )
    else:
        decoded = precomputed_grid
    times_plot, rcs_dbsm = _slice_time_window(
        decoded["times_datetime64"],
        decoded[field_name],
        start_time=start_time,
        end_time=end_time,
    )

    # Smoothing is now applied inside compute_rcs_grid on the SN time-series
    # before conversion to RCS. No further smoothing is required here.

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
    if overlay_predicted_range:
        if predicted_range_overlay == "ballistic":
            overlay_paths = get_fragment_fit_paths(fit_paths if fit_paths is not None else [fit_path])
            for overlay_fit_path in overlay_paths:
                fit_times = get_fragment_fit_time_track(
                    overlay_fit_path,
                    start_time=times_plot[0],
                    end_time=times_plot[-1],
                )
                if fit_times.size < 2:
                    continue
                range_info = get_fragment_fit_bragg_geometry(
                    tx=tx,
                    rx=rx,
                    time_dt64=fit_times,
                    fit_path=overlay_fit_path,
                )
                ax.plot(
                    fit_times,
                    range_info["propagation_range_km"],
                    color="white",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.45,
                    zorder=5,
                )
        elif predicted_range_overlay == "optical":
            for point_group in get_optical_fragment_range_points(tx=tx, rx=rx):
                point_times = n.asarray(point_group["times_datetime64"])
                if optical_fragment_delay_s != 0.0:
                    point_times = point_times + n.timedelta64(
                        int(round(float(optical_fragment_delay_s) * 1e9)),
                        "ns",
                    )
                point_ranges = n.asarray(point_group["propagation_range_km"], dtype=float)
                mask = n.isfinite(point_ranges) & (point_times >= times_plot[0]) & (point_times <= times_plot[-1])
                if not n.any(mask):
                    continue
                ax.scatter(
                    point_times[mask],
                    point_ranges[mask],
                    s=8.0,
                    marker="o",
                    facecolors="none",
                    edgecolors=optical_fragment_family_color(point_group["fragment_id"]),
                    linewidths=0.35,
                    zorder=6,
                )
        elif predicted_range_overlay not in ("none", None):
            raise ValueError(
                "predicted_range_overlay must be 'ballistic', 'optical', or 'none'."
            )
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Propagation range (km)")
    if title is not None:
        ax.set_title(title, pad=4)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(top=False, right=True)

    if show_aspect_axis:
        shifted_times_plot = times_plot + n.timedelta64(int(aspect_time_shift_s * 1e9), "ns")
        shifted_aspect = get_fragment_fit_bragg_geometry(
            tx=tx,
            rx=rx,
            time_dt64=shifted_times_plot,
            fit_path=fit_path,
        )["aspect_deg"]
        time_nums = mdates.date2num(times_plot.astype("datetime64[ms]"))
        if created_fig:
            fig.canvas.draw()
        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        finite = n.isfinite(time_nums) & n.isfinite(shifted_aspect)
        if n.count_nonzero(finite) >= 2:
            aspect_sorted_idx = n.argsort(shifted_aspect[finite])
            aspect_sorted = shifted_aspect[finite][aspect_sorted_idx]
            time_sorted = time_nums[finite][aspect_sorted_idx]
            tick_labels = []
            tick_locs = []
            for tick_val in aspect_tick_values:
                if aspect_sorted[0] <= tick_val <= aspect_sorted[-1]:
                    tick_locs.append(float(n.interp(tick_val, aspect_sorted, time_sorted)))
                    tick_labels.append(f"{tick_val:.0f}")
            if tick_locs:
                top_ax.set_xticks(tick_locs)
                top_ax.set_xticklabels(tick_labels)
        top_ax.set_xlabel("Aspect angle (deg)")
        top_ax.tick_params(direction="out")

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
    fit_path="ballistic_fit_sharedstart_1.h5",
):
    decoded = compute_rcs_grid(tx=tx, rx=rx)
    ridx = n.where((decoded["range_km"] > ymin) & (decoded["range_km"] < ymax))[0]
    if ridx.size == 0:
        raise ValueError("No range bins fall inside the requested ymin/ymax interval.")

    rcs_dbsm = decoded["rcs_dbsm"]
    rcs_aspect_db = n.max(rcs_dbsm[ridx, :], axis=0)

    aspect_info = get_fragment_fit_bragg_geometry(
        tx=tx,
        rx=rx,
        time_dt64=decoded["times_datetime64"],
        fit_path=fit_path,
    )

    with plt.rc_context(publication_rcparams()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)

        ax.plot(
            aspect_info["aspect_deg"],
            rcs_aspect_db,
            ".",
            markersize=2.5,
            label="$F_1$ fit",
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


def plot_peak_snr_gate_timeseries(
    tx="jruh",
    rx="bornim",
    start_time=None,
    end_time=None,
    fit_start_time=None,
    fit_end_time=None,
    ymin=100,
    ymax=600,
    ylimits_db=(0, 50),
    ax=None,
    show=True,
    output_filename="rcs_peak_gate_timeseries_single_column.pdf",
    title=None,
    precomputed_grid=None,
):
    decoded = compute_rcs_grid(tx=tx, rx=rx) if precomputed_grid is None else precomputed_grid
    times_plot, snr_db = _slice_time_window(
        decoded["times_datetime64"],
        decoded["sn_plus_n_over_n_db"],
        start_time=start_time,
        end_time=end_time,
    )
    _, sn_plus_n_over_n_linear = _slice_time_window(
        decoded["times_datetime64"],
        decoded["snr"],
        start_time=start_time,
        end_time=end_time,
    )
    gate_idx, gate_range_km, peak_snr_db = _select_peak_range_gate(
        decoded["range_km"],
        snr_db,
        ymin=ymin,
        ymax=ymax,
    )
    time_series_range_info = get_fragment_fit_tx_rx_ranges_km(tx=tx, rx=rx, time_dt64=times_plot)
    rcs_linear = sn_plus_n_over_n_to_rcs(
        sn_plus_n_over_n_linear[gate_idx, :].copy(),
        n.asarray(time_series_range_info["r_tx_km"], dtype=float) * 1e3,
        n.asarray(time_series_range_info["r_rx_km"], dtype=float) * 1e3,
    )
    rcs_linear = n.where(n.isfinite(rcs_linear) & (rcs_linear > 0.0), rcs_linear, n.nan)
    rcs_dbsm = 10.0 * n.log10(n.where(n.isfinite(rcs_linear) & (rcs_linear > 0.0), rcs_linear, 1e-12))
    fit_result = fit_exponential_efolding_time(
        times_plot,
        rcs_linear,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )
    peak_idx = int(n.nanargmax(snr_db[gate_idx, :]))
    peak_time_dt64 = times_plot[peak_idx]
    range_info = get_fragment_fit_tx_rx_ranges_km(tx=tx, rx=rx, time_dt64=peak_time_dt64)
    peak_rcs_dbsm = float(rcs_dbsm[peak_idx])
    peak_height_km = float(range_info["hgt_km"])
    peak_mean_free_path_m = mfp.mean_free_path_m(
        time_dt64=peak_time_dt64,
        lat_deg=float(range_info["lat_deg"]),
        lon_deg=float(range_info["lon_deg"]),
        alt_km=peak_height_km,
    )

    created_fig = ax is None
    if created_fig:
        with plt.rc_context(publication_rcparams()):
            fig, ax = plt.subplots(figsize=(4.1, 2.2), constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot(
        times_plot,
        rcs_dbsm,
        ".",
        color="black",
        markersize=1.5,
        linewidth=1.2,
        label=f"RCS",
    )
    ax.plot(
        fit_result["fit_times_datetime64"],
        10.0 * n.log10(n.maximum(fit_result["fit_model_linear"], 1e-12)),
        color="#cb181d",
        linewidth=1.3,
        linestyle="--",
        label="Exponential fit",
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_title(
        get_link_display_name(tx, rx) if title is None else title,
        pad=4,
    )
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(top=False, right=True)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
    if start_time is not None or end_time is not None:
        x0 = _coerce_datetime64(start_time) if start_time is not None else times_plot[0]
        x1 = _coerce_datetime64(end_time) if end_time is not None else times_plot[-1]
        ax.set_xlim(x0, x1)
    if ylimits_db is not None:
        ax.set_ylim(*ylimits_db)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.96,
        (
            f"Peak RCS: {peak_rcs_dbsm:.1f} dBsm\n"
            f"$h$ = {peak_height_km:.1f} km\n"
            f"$\\lambda_{{\\rm mfp}}$ = {mfp.format_mean_free_path(peak_mean_free_path_m)}\n"
            f"$\\tau$ = {fit_result['tau_s']:.2f} $\\pm$ {fit_result['tau_2sigma_s']:.2f} s (2$\\sigma$)\n"
            f"$R_{{\\rm tx}}$ = {range_info['r_tx_km']:.1f} km\n"
            f"$R_{{\\rm rx}}$ = {range_info['r_rx_km']:.1f} km"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 2.0},
    )

    if created_fig:
        fig.autofmt_xdate(rotation=30, ha="right")

    if created_fig and output_filename is not None:
        fig.savefig(output_filename, bbox_inches="tight")
        print(f"Saved peak-gate RCS PDF: {output_filename}")

    if created_fig and show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return {
        "times_datetime64": times_plot,
        "snr_db": snr_db[gate_idx, :],
        "rcs_dbsm": rcs_dbsm,
        "range_gate_index": gate_idx,
        "range_gate_km": gate_range_km,
        "peak_snr_db": peak_snr_db,
        "peak_rcs_dbsm": peak_rcs_dbsm,
        "peak_time_datetime64": peak_time_dt64,
        "range_info": range_info,
        "fit_result": fit_result,
        "figure": fig,
        "axes": ax,
    }


if __name__ == "__main__":
    plot_peak_snr_gate_timeseries(
        tx="kborn",
        rx="hagenow",
        start_time="2025-02-19T03:46:01",
        end_time="2025-02-19T03:46:12",
        fit_start_time="2025-02-19T03:46:05",
        fit_end_time="2025-02-19T03:46:07",
        ylimits_db=(10, 70),
    )

    plot_decoded(
        tx="kborn",
        rx="hagenow",
        start_time="2025-02-19T03:45:47",
        end_time="2025-02-19T03:46:26",
        ymin=215,
        ymax=330,
    )

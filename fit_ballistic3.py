import numpy as n
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.interpolate as sint
import scipy.optimize as so
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from pymsis import msis

OMEGA_EARTH = 7.2921150e-5  # rad/s
MU_EARTH = 3.986004418e14   # m^3/s^2
J2_EARTH = 1.08262668e-3
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
DEFAULT_MSIS_F107 = 150.0
DEFAULT_MSIS_F107A = 150.0
DEFAULT_MSIS_AP = 4.0


def load_recovered_fragments():
    """
    Load recovered-fragment coordinates from the sibling ground_reco.py file.
    """
    module_path = Path(__file__).with_name("ground_reco.py")
    spec = importlib.util.spec_from_file_location("ground_reco", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    ground_reco = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ground_reco)
    return [
        (frag_id, info["lat"], info["lon"])
        for frag_id, info in sorted(ground_reco.frags.items())
    ]


RECOVERED_FRAGMENTS = load_recovered_fragments()


def _sanitize_filename_token(value):
    token = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in token)


def build_fit_result_hdf5_path(fit_ids, filename_prefix="ballistic_fit"):
    fit_ids = list(fit_ids)
    if len(fit_ids) == 0:
        ids_part = "none"
    else:
        ids_part = "_".join(_sanitize_filename_token(fid) for fid in fit_ids)
    return Path(__file__).with_name(f"{filename_prefix}_{ids_part}.h5")


def _write_hdf5_item(group, key, value, path, skipped):
    import h5py

    if isinstance(value, dict):
        subgroup = group.create_group(key)
        for subkey, subvalue in value.items():
            _write_hdf5_item(
                subgroup,
                subkey,
                subvalue,
                f"{path}/{subkey}",
                skipped,
            )
        return

    if value is None:
        subgroup = group.create_group(key)
        subgroup.attrs["is_none"] = True
        return

    if callable(value):
        skipped.append(path)
        return

    if isinstance(value, (str, bytes, n.str_, n.bytes_)):
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(key, data=value, dtype=dtype)
        return

    if isinstance(value, (list, tuple)):
        arr = n.asarray(value)
    else:
        arr = n.asarray(value)

    if arr.dtype.kind in "biufc":
        group.create_dataset(key, data=arr)
        return

    if arr.dtype.kind in "SU":
        dtype = h5py.string_dtype(encoding="utf-8")
        if arr.ndim == 0:
            group.create_dataset(key, data=str(arr.item()), dtype=dtype)
        else:
            group.create_dataset(
                key,
                data=n.asarray(arr.tolist(), dtype=object),
                dtype=dtype,
            )
        return

    if arr.dtype.kind == "O":
        flat = arr.ravel()
        if all(isinstance(item, (str, bytes, n.str_, n.bytes_)) for item in flat):
            dtype = h5py.string_dtype(encoding="utf-8")
            if arr.ndim == 0:
                group.create_dataset(key, data=str(arr.item()), dtype=dtype)
            else:
                group.create_dataset(
                    key,
                    data=n.asarray(arr.tolist(), dtype=object),
                    dtype=dtype,
                )
            return

    skipped.append(path)


def save_result_to_hdf5(result, fit_ids, filename_prefix="ballistic_fit"):
    import h5py

    out_path = build_fit_result_hdf5_path(fit_ids, filename_prefix=filename_prefix)
    skipped = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    print(out_path)
    with h5py.File(out_path, "w") as h5:
#        h5.create_dataset(
 #           "fit_ids",
  #          data=n.asarray([str(fid) for fid in fit_ids], dtype=object),
   #         dtype=string_dtype,
    #    )
        for key, value in result.items():
            _write_hdf5_item(h5, key, value, key, skipped)

        if skipped:
            h5.create_dataset(
                "_skipped_keys",
                data=n.asarray(skipped, dtype=object),
                dtype=string_dtype,
            )

    return out_path


def _decode_hdf5_string_dataset(dataset):
    values = dataset[()]
    if isinstance(values, bytes):
        return values.decode("utf-8")
    if isinstance(values, str):
        return values
    values = n.asarray(values)
    if values.ndim == 0:
        item = values.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    decoded = []
    for item in values.tolist():
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return tuple(decoded)


def load_fit_initial_guess_from_hdf5(hdf5_path):
    import h5py

    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        fit_ids = None
        if "fit_ids" in h5:
            fit_ids = _decode_hdf5_string_dataset(h5["fit_ids"])

        return {
            "hdf5_path": str(hdf5_path),
            "fit_ids": fit_ids,
            "p0_hat_eci": n.asarray(h5["p0_hat_eci"][()], dtype=float),
            "v0_hat_eci": n.asarray(h5["v0_hat_eci"][()], dtype=float),
            "B0_hat": n.asarray(h5["B0_hat"][()], dtype=float),
        }


def infer_shared_root_fragment_id(fit_ids):
    fit_ids = tuple(str(fid) for fid in fit_ids)
    root_ids = []
    for fid in fit_ids:
        if fid in ("1", "2") and fid not in root_ids:
            root_ids.append(fid)

    if len(root_ids) == 1:
        return root_ids[0]

    if len(root_ids) == 0:
        raise ValueError(
            "Could not infer shared root fragment from fit_ids=%s; "
            "each chain must include exactly one of '1' or '2'."
            % (fit_ids,)
        )

    raise ValueError(
        "Ambiguous shared root fragment for fit_ids=%s; "
        "a chain may not include both '1' and '2'."
        % (fit_ids,)
    )


def prepare_fragment_fit_data(
    fragment_pos,
    fragment_pos_err,
    fragment_times,
    fit_ids,
    terminal_weight=1.0,
    terminal_weight_seconds=0.0,
):
    t = n.asarray(fragment_times, dtype=float).reshape(-1)
    pos_ecef = n.asarray(fragment_pos, dtype=float)
    pos_ecef_err = n.asarray(fragment_pos_err, dtype=float).reshape(-1)

    if pos_ecef.ndim != 2 or pos_ecef.shape[1] != 3:
        raise ValueError("fragment_pos must have shape (n, 3)")
    if t.shape[0] != pos_ecef.shape[0]:
        raise ValueError("fragment_times and fragment_pos must have the same length")
    if pos_ecef_err.shape[0] != pos_ecef.shape[0]:
        raise ValueError("fragment_pos_err and fragment_pos must have the same length")
    if t.size < 3:
        raise ValueError("Need at least three 3D measurements for the fit.")

    order = n.argsort(t, kind="mergesort")
    t = t[order]
    pos_ecef = pos_ecef[order, :]
    pos_ecef_err = pos_ecef_err[order]

    p0_guess_ecef = pos_ecef[0, :]
    dt_obs = t[-1] - t[0]
    if dt_obs <= 0.0:
        raise ValueError("Need observations spanning a non-zero time interval.")
    v0_guess_ecef = estimate_initial_velocity_ecef(t, pos_ecef)

    pos_eci = ecef_to_eci_position(pos_ecef, t)
    residual_weights = n.ones(t.shape, dtype=float)
    terminal_weight = float(terminal_weight)
    terminal_weight_seconds = float(terminal_weight_seconds)
    if terminal_weight > 1.0 and terminal_weight_seconds > 0.0:
        terminal_start = float(t[-1] - terminal_weight_seconds)
        ramp = n.clip((t - terminal_start) / terminal_weight_seconds, 0.0, 1.0)
        residual_weights = 1.0 + (terminal_weight - 1.0) * ramp

    density_profile = build_msis_density_profile(t, pos_ecef)
    p0_guess_eci, v0_guess_eci = ecef_to_eci_state(
        p0_guess_ecef,
        v0_guess_ecef,
        t[0],
    )

    fit_ids = tuple(str(fid) for fid in fit_ids)
    shared_start_id = infer_shared_root_fragment_id(fit_ids)

    return {
        "times_unix": t,
        "pos_ecef": pos_ecef,
        "pos_ecef_err": pos_ecef_err,
        "pos_eci": pos_eci,
        "residual_weights": residual_weights,
        "fit_ids": fit_ids,
        "root_fragment_id": shared_start_id,
        "shared_start_id": shared_start_id,
        "p0_guess_ecef": p0_guess_ecef,
        "v0_guess_ecef": v0_guess_ecef,
        "p0_guess_eci": p0_guess_eci,
        "v0_guess_eci": v0_guess_eci,
        "density_profile": density_profile,
    }

def get_msis_density(times_unix, lat_deg, lon_deg, alt_m):
    """
    Evaluate MSIS total mass density for each sample.
    Returns rho_a in kg/m^3.
    """
    times_dt64 = times_unix.astype("datetime64[s]")

    rho_a = n.full(len(times_unix), n.nan)

    for j in range(len(times_unix)):
        data = msis.run(
            n.array([times_dt64[j]]),
            n.array([lat_deg[j]]),
            n.array([lon_deg[j]]),
            n.array([alt_m[j] / 1e3]),   # km
            f107s=DEFAULT_MSIS_F107,
            f107as=DEFAULT_MSIS_F107A,
            aps=n.full((1, 7), DEFAULT_MSIS_AP, dtype=float),
            geomagnetic_activity=-1,
        )

        arr = n.asarray(data)

        # pymsis total mass density is typically the first species/output entry.
        # Squeeze to make indexing robust across wrapper return shapes.
        arr = n.squeeze(arr)
        rho_a[j] = arr[0] if n.ndim(arr) > 0 else arr

    return rho_a


def _extract_msis_total_density(data):
    arr = n.asarray(data)
    arr = n.squeeze(arr)
    if arr.ndim == 0:
        return n.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return n.asarray([float(arr[0])], dtype=float)
    return n.asarray(arr[..., 0], dtype=float)


def circular_mean_deg(angle_deg):
    angle_rad = n.deg2rad(n.asarray(angle_deg, dtype=float))
    return float(n.rad2deg(n.angle(n.mean(n.exp(1j * angle_rad)))))


def estimate_initial_velocity_ecef(times_unix, pos_ecef, max_initial_span_s=12.0, min_points=6):
    """
    Estimate the initial velocity from the first few position measurements.

    A whole-track first-to-last velocity can be biased when the fragment is
    decelerating or when a merged path contains parent/child segments. For the
    trajectory fit, the state vector is defined at the beginning of the fitted
    interval, so the initial slope is the better starting point.
    """
    t = n.asarray(times_unix, dtype=float).reshape(-1)
    pos = n.asarray(pos_ecef, dtype=float)
    if t.size < 2 or pos.shape[0] != t.size:
        raise ValueError("Need matching time and position samples.")

    dt_total = t[-1] - t[0]
    if dt_total <= 0.0:
        raise ValueError("Need observations spanning a non-zero time interval.")

    use = t <= t[0] + float(max_initial_span_s)
    if n.count_nonzero(use) < int(min_points):
        use = n.zeros(t.shape, dtype=bool)
        use[: min(int(min_points), t.size)] = True

    if n.count_nonzero(use) < 2:
        return (pos[-1, :] - pos[0, :]) / dt_total

    t_fit = t[use] - t[0]
    pos_fit = pos[use, :]
    design = n.column_stack((n.ones(t_fit.shape), t_fit))
    coeffs, _, _, _ = n.linalg.lstsq(design, pos_fit, rcond=None)
    return n.asarray(coeffs[1, :], dtype=float)


def build_msis_density_profile(times_unix, pos_ecef, n_altitude=512, min_top_alt_m=300e3, alt_pad_m=50e3):
    """
    Build a 1D density profile rho(h) using a single MSIS call at the mean
    measurement time and mean geodetic latitude/longitude.
    """
    times_unix = n.asarray(times_unix, dtype=float).reshape(-1)
    pos_ecef = n.asarray(pos_ecef, dtype=float)
    if times_unix.size == 0 or pos_ecef.shape[0] == 0:
        raise ValueError("Need measurements to build the MSIS density profile.")

    lat_rad, lon_rad, hgt_m = ecef_to_geodetic_wgs84(pos_ecef)
    lat_deg = n.rad2deg(lat_rad)
    lon_deg = n.rad2deg(lon_rad)

    ref_time_unix = float(n.mean(times_unix))
    ref_lat_deg = float(n.mean(lat_deg))
    ref_lon_deg = circular_mean_deg(lon_deg)
    top_alt_m = float(max(min_top_alt_m, n.nanmax(hgt_m) + alt_pad_m))

    altitude_grid_m = n.linspace(0.0, top_alt_m, int(n_altitude), dtype=float)
    time_grid = n.full_like(altitude_grid_m, ref_time_unix, dtype=float).astype("datetime64[s]")
    lat_grid = n.full_like(altitude_grid_m, ref_lat_deg, dtype=float)
    lon_grid = n.full_like(altitude_grid_m, ref_lon_deg, dtype=float)
    f107_grid = n.full(altitude_grid_m.size, DEFAULT_MSIS_F107, dtype=float)
    f107a_grid = n.full(altitude_grid_m.size, DEFAULT_MSIS_F107A, dtype=float)
    ap_grid = n.full((altitude_grid_m.size, 7), DEFAULT_MSIS_AP, dtype=float)

    data = msis.run(
        time_grid,
        lat_grid,
        lon_grid,
        altitude_grid_m / 1e3,
        f107s=f107_grid,
        f107as=f107a_grid,
        aps=ap_grid,
        geomagnetic_activity=-1,
    )
    rho_grid = _extract_msis_total_density(data)
    if rho_grid.shape[0] != altitude_grid_m.shape[0]:
        rho_grid = n.asarray(rho_grid).reshape(-1)
    if rho_grid.shape[0] != altitude_grid_m.shape[0]:
        raise ValueError("Unexpected MSIS output shape for density profile.")

    rho_grid = n.maximum(rho_grid, 1e-30)
    log_rho_interp = sint.interp1d(
        altitude_grid_m,
        n.log(rho_grid),
        kind="linear",
        bounds_error=False,
        fill_value=(float(n.log(rho_grid[0])), float(n.log(rho_grid[-1]))),
        assume_sorted=True,
    )

    def density_interp(altitude_m_query):
        altitude_arr = n.asarray(altitude_m_query, dtype=float)
        altitude_clip = n.clip(altitude_arr, altitude_grid_m[0], altitude_grid_m[-1])
        rho_query = n.exp(log_rho_interp(altitude_clip))
        if altitude_arr.ndim == 0:
            return float(rho_query)
        return n.asarray(rho_query, dtype=float)

    return {
        "reference_time_unix": ref_time_unix,
        "reference_lat_deg": ref_lat_deg,
        "reference_lon_deg": ref_lon_deg,
        "altitude_grid_m": altitude_grid_m,
        "rho_grid_kg_m3": rho_grid,
        "interp": density_interp,
    }

def gmst_angle(unix_time):
    """
    Greenwich mean sidereal angle in radians.
    """
    unix_time = n.asarray(unix_time, dtype=float)
    jd = unix_time / 86400.0 + 2440587.5
    t_ut1 = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t_ut1**2
        - t_ut1**3 / 38710000.0
    )
    return n.deg2rad(n.mod(gmst_deg, 360.0))


def ecef_to_eci_position(pos_ecef, unix_time):
    """
    Convert ECEF position vector(s) to ECI with a GMST rotation.
    """
    pos_ecef = n.asarray(pos_ecef, dtype=float)
    theta = gmst_angle(unix_time)
    c = n.cos(theta)
    s = n.sin(theta)

    x = pos_ecef[..., 0]
    y = pos_ecef[..., 1]
    z = pos_ecef[..., 2]

    x_eci = c * x - s * y
    y_eci = s * x + c * y
    return n.stack((x_eci, y_eci, z), axis=-1)


def eci_to_ecef_position(pos_eci, unix_time):
    """
    Convert ECI position vector(s) back to ECEF coordinates.
    """
    pos_eci = n.asarray(pos_eci, dtype=float)
    theta = gmst_angle(unix_time)
    c = n.cos(theta)
    s = n.sin(theta)

    x = pos_eci[..., 0]
    y = pos_eci[..., 1]
    z = pos_eci[..., 2]

    x_ecef = c * x + s * y
    y_ecef = -s * x + c * y
    return n.stack((x_ecef, y_ecef, z), axis=-1)


def ecef_to_eci_state(pos_ecef, vel_ecef, unix_time, omega=OMEGA_EARTH):
    """
    Convert an ECEF position/velocity state vector to ECI.
    """
    pos_ecef = n.asarray(pos_ecef, dtype=float)
    vel_ecef = n.asarray(vel_ecef, dtype=float)
    omega_vec = n.array([0.0, 0.0, float(omega)])

    pos_eci = ecef_to_eci_position(pos_ecef, unix_time)
    vel_eci = ecef_to_eci_position(
        vel_ecef + n.cross(omega_vec, pos_ecef),
        unix_time,
    )
    return pos_eci, vel_eci


def ecef_to_geodetic_wgs84(pos_ecef):
    """
    Convert ECEF position vector(s) to geodetic latitude, longitude, and
    ellipsoidal height above the WGS84 reference ellipsoid.
    """
    pos_ecef = n.asarray(pos_ecef, dtype=float)
    x = pos_ecef[..., 0]
    y = pos_ecef[..., 1]
    z = pos_ecef[..., 2]

    lon = n.arctan2(y, x)
    p = n.sqrt(x**2 + y**2)

    theta = n.arctan2(z * WGS84_A, p * WGS84_B)
    sin_theta = n.sin(theta)
    cos_theta = n.cos(theta)

    lat = n.arctan2(
        z + WGS84_EP2 * WGS84_B * sin_theta**3,
        p - WGS84_E2 * WGS84_A * cos_theta**3,
    )

    sin_lat = n.sin(lat)
    cos_lat = n.cos(lat)
    N = WGS84_A / n.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    h = n.where(
        n.abs(cos_lat) > 1e-12,
        p / cos_lat - N,
        z / n.where(n.abs(sin_lat) > 1e-12, sin_lat, 1.0) - N * (1.0 - WGS84_E2),
    )

    return lat, lon, h


def eci_to_geodetic(pos_eci, unix_time):
    """
    Convert ECI position vector(s) to geodetic latitude/longitude in degrees
    and ellipsoidal height above the WGS84 reference ellipsoid in meters.
    """
    pos_ecef = eci_to_ecef_position(pos_eci, unix_time)
    lat, lon, h = ecef_to_geodetic_wgs84(pos_ecef)
    return n.rad2deg(lat), n.rad2deg(lon), h


def gravity_accel_eci_j2(pos_eci, mu=MU_EARTH, j2=J2_EARTH, r_eq=WGS84_A):
    """
    Gravitational acceleration in ECI including the Earth's J2 term.

    Parameters
    ----------
    pos_eci : array_like
        Position vector(s) in ECI coordinates, shape (..., 3), in meters.
    """
    pos_eci = n.asarray(pos_eci, dtype=float)
    x = pos_eci[..., 0]
    y = pos_eci[..., 1]
    z = pos_eci[..., 2]

    r2 = x**2 + y**2 + z**2
    r = n.sqrt(r2)
    r3 = r2 * r

    a_kepler = -mu * pos_eci / r3[..., None]

    z2_over_r2 = z**2 / r2
    j2_prefactor = 1.5 * j2 * mu * r_eq**2 / (r**5)
    a_j2 = j2_prefactor[..., None] * n.stack(
        (
            x * (5.0 * z2_over_r2 - 1.0),
            y * (5.0 * z2_over_r2 - 1.0),
            z * (5.0 * z2_over_r2 - 3.0),
        ),
        axis=-1,
    )

    return a_kepler + a_j2


def atmosphere_velocity_eci(pos_eci, omega=OMEGA_EARTH):
    """
    Velocity of an atmosphere assumed to co-rotate rigidly with the Earth,
    expressed in ECI coordinates.
    """
    pos_eci = n.asarray(pos_eci, dtype=float)
    omega_vec = n.array([0.0, 0.0, float(omega)])
    return n.cross(omega_vec, pos_eci)


def build_era5_request_area(pos_ecef, pad_deg=2.0):
    lat_rad, lon_rad, _ = ecef_to_geodetic_wgs84(pos_ecef)
    lat_deg = n.rad2deg(n.asarray(lat_rad, dtype=float))
    lon_deg = n.rad2deg(n.asarray(lon_rad, dtype=float))
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0

    north = float(min(90.0, n.nanmax(lat_deg) + pad_deg))
    south = float(max(-90.0, n.nanmin(lat_deg) - pad_deg))
    west = float(max(-180.0, n.nanmin(lon_deg) - pad_deg))
    east = float(min(180.0, n.nanmax(lon_deg) + pad_deg))
    return [north, west, south, east]


def load_reanalysis_wind_model(
    times_unix,
    pos_ecef,
    max_time_ahead=3600.0,
    area_pad_deg=2.0,
    verbose=0,
):
    info = {
        "type": "corotating_fallback",
        "area": None,
        "start_time_unix": float(n.min(times_unix)),
        "end_time_unix": float(n.max(times_unix)),
    }

    try:
        from era5wind import load_or_download_era5, load_or_download_era5_model_levels
    except Exception as exc:
        info["error"] = f"Could not import era5wind: {exc}"
        if verbose > 0:
            print("era5wind_import_error", info["error"])
        return None, info

    area = build_era5_request_area(pos_ecef, pad_deg=area_pad_deg)
    request_start = float(n.min(times_unix) - 3600.0)
    request_end = float(n.max(times_unix) + max_time_ahead + 3600.0)
    info["area"] = area
    info["start_time_unix"] = request_start
    info["end_time_unix"] = request_end

    cache_dir = Path(__file__).with_name("data")
    time_tag = datetime.fromtimestamp(float(n.mean(times_unix)), tz=timezone.utc).strftime("%Y%m%d_%H")
    model_prefix = cache_dir / f"era5_model_levels_{time_tag}"
    pressure_path = cache_dir / f"era5_pressure_levels_{time_tag}.nc"

    try:
        wind_model = load_or_download_era5_model_levels(
            target_prefix=model_prefix,
            start_time_unix=request_start,
            end_time_unix=request_end,
            area=area,
            overwrite=False,
        )
        info.update(
            {
                "type": "era5_model_levels",
                "cache_prefix": str(model_prefix),
            }
        )
        if verbose > 0:
            print("wind_model", info["type"], info["cache_prefix"])
        return wind_model, info
    except Exception as exc:
        info["model_level_error"] = str(exc)
        if verbose > 0:
            print("era5_model_level_error", exc)

    try:
        wind_model = load_or_download_era5(
            target_path=pressure_path,
            start_time_unix=request_start,
            end_time_unix=request_end,
            area=area,
            overwrite=False,
        )
        info.update(
            {
                "type": "era5_pressure_levels",
                "cache_path": str(pressure_path),
            }
        )
        if verbose > 0:
            print("wind_model", info["type"], info["cache_path"])
        return wind_model, info
    except Exception as exc:
        info["pressure_level_error"] = str(exc)
        if verbose > 0:
            print("era5_pressure_level_error", exc)

    return None, info


class CachedTrajectoryWindModel:
    """
    Atmosphere-velocity model obtained by sampling ERA5 along a nominal
    trajectory and interpolating the resulting ECI atmosphere velocity in time.
    """

    def __init__(self, times_unix, atmosphere_velocity_eci_samples, info=None):
        times_unix = n.asarray(times_unix, dtype=float).reshape(-1)
        atmosphere_velocity_eci_samples = n.asarray(
            atmosphere_velocity_eci_samples,
            dtype=float,
        )

        if times_unix.size == 0:
            raise ValueError("Need at least one wind sample.")
        if atmosphere_velocity_eci_samples.shape != (times_unix.size, 3):
            raise ValueError("Wind samples must have shape (n, 3).")

        order = n.argsort(times_unix, kind="mergesort")
        self.times_unix = times_unix[order]
        self.atmosphere_velocity_eci_samples = atmosphere_velocity_eci_samples[order, :]
        self.info = {} if info is None else dict(info)
        self._interp = sint.interp1d(
            self.times_unix,
            self.atmosphere_velocity_eci_samples,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(
                self.atmosphere_velocity_eci_samples[0, :],
                self.atmosphere_velocity_eci_samples[-1, :],
            ),
            assume_sorted=True,
        )

    def atmosphere_velocity_eci(self, lat_deg, lon_deg, hgt_m, unix_time, omega=OMEGA_EARTH):
        return n.asarray(self._interp(float(unix_time)), dtype=float)


def build_nominal_wind_sampling_track(result, max_time_ahead=3600.0):
    model = result["model"]
    t_model = n.asarray(model["times_model"], dtype=float)
    pos_model = n.asarray(model["pos_eci"], dtype=float)
    lat_model = n.asarray(model["lat_deg"], dtype=float)
    lon_model = n.asarray(model["lon_deg"], dtype=float)
    hgt_model = n.asarray(model["hgt_m"], dtype=float)

    t_start = float(n.max(result["times_unix"]))
    dt_use = float(result["dt_model"])

    p_start = n.asarray(model["pos_eci_interp"](t_start), dtype=float)
    v_start = n.asarray(model["vel_eci_interp"](t_start), dtype=float)
    B_start = n.log10(float(model["B_interp"](t_start)))

    extension = propagate(
        p_start,
        v_start,
        n.array([t_start, t_start + float(max_time_ahead)], dtype=float),
        B_start,
        dt=dt_use,
        fixed_B=B_start,
        start_time=t_start,
        stop_at_ground=True,
        density_profile=model.get("density_profile"),
        wind_model=model.get("wind_model"),
    )

    t_extension = n.asarray(extension["times_model"], dtype=float)
    pos_extension = n.asarray(extension["pos_eci"], dtype=float)
    lat_extension = n.asarray(extension["lat_deg"], dtype=float)
    lon_extension = n.asarray(extension["lon_deg"], dtype=float)
    hgt_extension = n.asarray(extension["hgt_m"], dtype=float)

    if t_extension.size > 0:
        t_track = n.concatenate((t_model, t_extension[1:]))
        pos_track = n.vstack((pos_model, pos_extension[1:, :]))
        lat_track = n.concatenate((lat_model, lat_extension[1:]))
        lon_track = n.concatenate((lon_model, lon_extension[1:]))
        hgt_track = n.concatenate((hgt_model, hgt_extension[1:]))
    else:
        t_track = t_model
        pos_track = pos_model
        lat_track = lat_model
        lon_track = lon_model
        hgt_track = hgt_model

    return {
        "times_unix": t_track,
        "pos_eci": pos_track,
        "lat_deg": lat_track,
        "lon_deg": lon_track,
        "hgt_m": hgt_track,
        "ground_reached": bool(n.any(hgt_track <= 0.0)),
    }


def build_cached_era5_wind_model_from_result(result, max_time_ahead=3600.0, verbose=0):
    track = build_nominal_wind_sampling_track(result, max_time_ahead=max_time_ahead)
    pos_ecef = eci_to_ecef_position(track["pos_eci"], track["times_unix"])
    source_wind_model, source_info = load_reanalysis_wind_model(
        track["times_unix"],
        pos_ecef,
        max_time_ahead=0.0,
        verbose=verbose,
    )

    info = {
        "type": "corotating_fallback",
        "sample_count": int(track["times_unix"].size),
        "time_start_unix": float(track["times_unix"][0]),
        "time_end_unix": float(track["times_unix"][-1]),
        "ground_reached_in_profile": bool(track["ground_reached"]),
        "source": source_info,
    }

    if source_wind_model is None:
        return None, info

    atmosphere_velocity_samples = n.empty((track["times_unix"].size, 3), dtype=float)
    for i, tnow in enumerate(track["times_unix"]):
        try:
            sample = n.asarray(
                source_wind_model.atmosphere_velocity_eci(
                    track["lat_deg"][i],
                    track["lon_deg"][i],
                    max(float(track["hgt_m"][i]), 0.0),
                    float(tnow),
                ),
                dtype=float,
            )
            if sample.shape != (3,) or not n.all(n.isfinite(sample)):
                raise ValueError("Non-finite ERA5 atmosphere velocity sample.")
            atmosphere_velocity_samples[i, :] = sample
        except Exception:
            atmosphere_velocity_samples[i, :] = atmosphere_velocity_eci(track["pos_eci"][i, :])

    info["type"] = "era5_time_profile_interp"
    cached_model = CachedTrajectoryWindModel(
        track["times_unix"],
        atmosphere_velocity_samples,
        info=info,
    )
    return cached_model, info


def sigmoid(x):
    return 1.0 / (1.0 + n.exp(-x))


def ballistic_node_times(times, dt_model, n_nodes):
    times = n.asarray(times, dtype=float).reshape(-1)
    n_nodes = int(n_nodes)
    if times.size == 0:
        raise ValueError("Need at least one time sample to define ballistic node times.")
    if n_nodes < 2:
        raise ValueError("Need at least two ballistic coefficient node points.")
    return n.linspace(
        float(n.min(times) - 2.0 * dt_model),
        float(n.max(times) + 2.0 * dt_model),
        n_nodes,
        dtype=float,
    )


def linear_interpolation_weights(query_times, knot_times):
    query_times = n.asarray(query_times, dtype=float).reshape(-1)
    knot_times = n.asarray(knot_times, dtype=float).reshape(-1)
    if knot_times.size < 2:
        raise ValueError("Need at least two knot times for interpolation weights.")

    weights = n.zeros((query_times.size, knot_times.size), dtype=float)
    for i, t_query in enumerate(query_times):
        if t_query <= knot_times[0]:
            weights[i, 0] = 1.0
            continue
        if t_query >= knot_times[-1]:
            weights[i, -1] = 1.0
            continue

        i0 = int(n.searchsorted(knot_times, t_query, side="right") - 1)
        i0 = max(0, min(i0, knot_times.size - 2))
        i1 = i0 + 1
        span = float(knot_times[i1] - knot_times[i0])
        if span <= 0.0:
            weights[i, i0] = 1.0
            continue
        alpha = float((t_query - knot_times[i0]) / span)
        weights[i, i0] = 1.0 - alpha
        weights[i, i1] = alpha

    return weights




def build_ballistic_profile(times, B0, fr_raw):
    times = n.asarray(times, dtype=float)
    if times.size == 0:
        return n.array([], dtype=float), n.array([], dtype=float)

    fr_raw = n.asarray(fr_raw, dtype=float)
    fr_scale = sigmoid(fr_raw)

    B_values = [float(B0)]
    B_now = float(B0)
    for scale in fr_scale:
        B_now *= scale
        B_values.append(B_now)
    B_values = n.asarray(B_values, dtype=float)

    if times[-1] <= times[0]:
        return n.full(times.shape, B_values[0], dtype=float), fr_scale

    knot_times = n.linspace(times[0], times[-1], B_values.size)
    return n.interp(times, knot_times, B_values), fr_scale


def evaluate_model_at_times(times_model, values_model, times_eval):
    times_model = n.asarray(times_model, dtype=float)
    values_model = n.asarray(values_model, dtype=float)
    times_eval = n.asarray(times_eval, dtype=float)

    cols = []
    for i in range(values_model.shape[1]):
        cols.append(n.interp(times_eval, times_model, values_model[:, i]))
    return n.column_stack(cols)


def state_to_geodetic_density(pos_eci, unix_time, density_profile=None):
    lat, lon, hgt = eci_to_geodetic(pos_eci, unix_time)
    if density_profile is None:
        hgt_msis = float(n.clip(hgt, 0.0, 300e3))
        rho_a = get_msis_density(
            n.array([unix_time], dtype=float),
            n.array([lat], dtype=float),
            n.array([lon], dtype=float),
            n.array([hgt_msis], dtype=float),
        )[0]
    else:
        rho_a = density_profile["interp"](max(float(hgt), 0.0))
    return float(lat), float(lon), float(hgt), float(rho_a)


def drag_state_eci(pos_eci, vel_eci, unix_time, B_now, density_profile=None, wind_model=None):
    """
    Evaluate local atmospheric state, drag acceleration, and specific drag
    power at a single ECI state.
    """
    lat, lon, hgt, rho_a = state_to_geodetic_density(pos_eci, unix_time, density_profile=density_profile)
    if wind_model is None:
        atmosphere_vel_eci = atmosphere_velocity_eci(pos_eci)
    else:
        try:
            atmosphere_vel_eci = wind_model.atmosphere_velocity_eci(
                lat,
                lon,
                max(float(hgt), 0.0),
                unix_time,
            )
        except Exception:
            atmosphere_vel_eci = atmosphere_velocity_eci(pos_eci)
    atmosphere_vel_eci = n.asarray(atmosphere_vel_eci, dtype=float)
    if atmosphere_vel_eci.shape != (3,) or not n.all(n.isfinite(atmosphere_vel_eci)):
        atmosphere_vel_eci = atmosphere_velocity_eci(pos_eci)
    v_rel = vel_eci - atmosphere_vel_eci
    vmag_rel = n.linalg.norm(v_rel)

    if vmag_rel > 0.0:
        v_rel_unit = v_rel / vmag_rel
        a_drag = -0.5 * rho_a * v_rel_unit * (vmag_rel**2) * B_now
    else:
        a_drag = n.zeros(3, dtype=float)

    # Positive values indicate drag-driven loss of specific kinetic energy
    # relative to the co-rotating atmosphere.
    specific_energy_loss_rate = float(-n.dot(v_rel, a_drag))
    speed = float(n.linalg.norm(vel_eci))
    return lat, lon, hgt, rho_a, v_rel, a_drag, specific_energy_loss_rate, speed, float(vmag_rel)


def ballistic_coefficient_uncertainty(result, times_query, B_values=None):
    parameter_covariance = result.get("parameter_covariance")
    if parameter_covariance is None:
        return None

    covariance = n.asarray(parameter_covariance, dtype=float)
    logB_hat = n.asarray(result["B0_hat"], dtype=float).reshape(-1)
    n_nodes = logB_hat.size
    if covariance.shape[0] < 6 + n_nodes or covariance.shape[1] < 6 + n_nodes:
        return None

    times_query = n.asarray(times_query, dtype=float)
    t_meas = n.asarray(result["times_unix"], dtype=float)
    dt_model = float(result["dt_model"])
    node_times = result.get("B_node_times_unix")
    if node_times is None:
        node_times = ballistic_node_times(t_meas, dt_model, n_nodes)
    else:
        node_times = n.asarray(node_times, dtype=float).reshape(-1)
        if node_times.size != n_nodes:
            node_times = ballistic_node_times(t_meas, dt_model, n_nodes)

    weights = linear_interpolation_weights(times_query, node_times)
    covariance_logB = covariance[6:6 + n_nodes, 6:6 + n_nodes]
    logB_variance = n.einsum("ni,ij,nj->n", weights, covariance_logB, weights)
    logB_sigma = n.sqrt(n.maximum(logB_variance, 0.0))

    if B_values is None:
        B_values = 10.0 ** (weights @ logB_hat)
    else:
        B_values = n.asarray(B_values, dtype=float)

    return n.log(10.0) * B_values * logB_sigma


def plot_sparse_errorbars(ax, x, y, yerr, max_points=24, **kwargs):
    x = n.asarray(x, dtype=float)
    y = n.asarray(y, dtype=float)
    yerr = n.asarray(yerr, dtype=float)

    mask = n.isfinite(x) & n.isfinite(y) & n.isfinite(yerr) & (y > 0.0) & (yerr >= 0.0)
    if not n.any(mask):
        return

    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    if x.size > max_points:
        idx = n.linspace(0, x.size - 1, max_points, dtype=int)
        x = x[idx]
        y = y[idx]
        yerr = yerr[idx]

    lower = n.minimum(yerr, 0.95 * y)
    upper = yerr
    ax.errorbar(
        x,
        y,
        yerr=n.vstack((lower, upper)),
        fmt="none",
        **kwargs,
    )


def style_publication_axis(ax, tick_labelsize=11):
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35, color="0.55")
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=tick_labelsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def load_fragment_plot_context():
    try:
        import plot_fragments as pf
    except Exception:
        return None, None

    try:
        _, _, _, _, _, fragment_geo_pos, fragment_times = pf.get_fragments()
    except Exception:
        return None, None

    return fragment_geo_pos, fragment_times


def get_fragment_plot_context(result):
    context_geo = result.get("plot_context_fragment_geo_pos")
    context_times = result.get("plot_context_fragment_times")
    if context_geo is None or context_times is None:
        loaded_geo, loaded_times = load_fragment_plot_context()
        if context_geo is None:
            context_geo = loaded_geo
        if context_times is None:
            context_times = loaded_times
    return context_geo, context_times


def plot_all_fragment_background(
    ax,
    context_geo,
    context_times=None,
    t0=None,
    panel="lonalt",
    B_interp=None,
):
    if context_geo is None:
        return

    background_labeled = False
    for i, geo in enumerate(context_geo):
        geo = n.asarray(geo, dtype=float)
        if geo.size == 0:
            continue

        label = "All fragment measurements" if not background_labeled else None

        if panel == "map":
            x = geo[:, 1]
            y = geo[:, 0]
        elif panel == "lonalt":
            x = geo[:, 1]
            y = geo[:, 2] / 1e3
        elif panel == "time":
            if context_times is None or t0 is None or i >= len(context_times):
                continue
            times = n.asarray(context_times[i], dtype=float)
            if times.size == 0:
                continue
            x = times - float(t0)
            y = geo[:, 2] / 1e3
        elif panel == "B":
            if context_times is None or t0 is None or B_interp is None or i >= len(context_times):
                continue
            times = n.asarray(context_times[i], dtype=float)
            if times.size == 0:
                continue
            B_values = n.asarray(B_interp(times), dtype=float)
            mask = n.isfinite(times) & n.isfinite(B_values) & (B_values > 0.0)
            if not n.any(mask):
                continue
            x = times[mask] - float(t0)
            y = B_values[mask]
        else:
            raise ValueError(f"Unknown panel '{panel}'")

        ax.plot(
            x,
            y,
            ".",
            color="0.82",
            markersize=2.5,
            alpha=0.45,
            zorder=1,
            rasterized=True,
            label=label,
        )
        background_labeled = True


def sample_fit_ensemble(result, n_samples=100, random_seed=0):
    covariance = result.get("parameter_covariance")
    if covariance is None:
        return []

    xhat = n.concatenate(
        (
            n.asarray(result["p0_hat_eci"], dtype=float),
            n.asarray(result["v0_hat_eci"], dtype=float),
            n.asarray(result["B0_hat"], dtype=float),
        )
    )
    covariance = n.asarray(covariance, dtype=float)
    if covariance.shape != (xhat.size, xhat.size):
        return []

    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = n.linalg.eigh(covariance)
    eigvals = n.maximum(eigvals, 0.0)
    positive = eigvals > 0.0
    if not n.any(positive):
        return []

    transform = eigvecs[:, positive] @ n.diag(n.sqrt(eigvals[positive]))
    rng = n.random.default_rng(random_seed)
    t = n.asarray(result["times_unix"], dtype=float)
    dt_model = float(result["dt_model"])
    density_profile = result["model"].get("density_profile")
    wind_model = result["model"].get("wind_model")

    samples = []
    max_attempts = max(5 * int(n_samples), 200)
    for _ in range(max_attempts):
        trial_x = xhat + transform @ rng.standard_normal(transform.shape[1])
        try:
            trial_model = build_model_from_parameters(
                trial_x,
                t,
                dt_model,
                density_profile=density_profile,
                wind_model=wind_model,
            )
        except Exception:
            continue

        trial_sample = {"model": trial_model, "impact": None}
        if result.get("impact") is not None:
            try:
                trial_sample["impact"] = extrapolate_best_fit_to_ground(
                    {
                        "model": trial_model,
                        "times_unix": t,
                        "dt_model": dt_model,
                    }
                )
            except Exception:
                trial_sample["impact"] = None

        samples.append(trial_sample)
        if len(samples) >= int(n_samples):
            break

    return samples


def plot_ballistic_fit(result, show=True):
    t_meas = n.asarray(result["times_unix"], dtype=float)
    t0 = n.min(t_meas)

    lat_meas, lon_meas, hgt_meas = eci_to_geodetic(result["pos_eci"], t_meas)

    model = result["model"]
    order = n.argsort(model["times_model"])
    t_model = model["times_model"][order]
    lat_model = model["lat_deg"][order]
    lon_model = model["lon_deg"][order]
    hgt_model = model["hgt_m"][order]
    B_model = model["B_model"][order]
    impact = result.get("impact")
    impact_uncertainty = result.get("impact_uncertainty")
    context_geo, context_times = get_fragment_plot_context(result)
    fit_samples = sample_fit_ensemble(result, n_samples=100, random_seed=0)

    fig, axes = plt.subplot_mosaic(
        [["map", "time"], ["B", "lonalt"]],
        figsize=(12.0, 8.8),
        constrained_layout=True,
    )

    ax_map = axes["map"]
    ax_time = axes["time"]
    ax_B = axes["B"]
    ax_lonalt = axes["lonalt"]

    sample_color = "#fcae91"
    model_color = "#cb181d"
    measurement_color = "0.45"
    axis_label_fontsize = 16
    tick_label_fontsize = 14
    B_tick_label_fontsize = 14
    legend_fontsize = 13
    annotation_fontsize = 10.5
    best_fit_zorder = 14
    extrapolated_zorder = 15
    measurement_zorder = 9

    for ax in (ax_map, ax_time, ax_B, ax_lonalt):
        ax.set_facecolor("white")

    plot_all_fragment_background(ax_map, context_geo, panel="map")
    plot_all_fragment_background(ax_time, context_geo, context_times=context_times, t0=t0, panel="time")
    plot_all_fragment_background(ax_lonalt, context_geo, panel="lonalt")

    sample_label_used = False
    for sample in fit_samples:
        sample_model = sample["model"]
        sample_order = n.argsort(sample_model["times_model"])
        sample_t = n.asarray(sample_model["times_model"][sample_order], dtype=float)
        sample_lat = n.asarray(sample_model["lat_deg"][sample_order], dtype=float)
        sample_lon = n.asarray(sample_model["lon_deg"][sample_order], dtype=float)
        sample_hgt = n.asarray(sample_model["hgt_m"][sample_order], dtype=float) / 1e3
        sample_B = n.asarray(sample_model["B_model"][sample_order], dtype=float)

        sample_label = "Uncertainty samples" if not sample_label_used else None
        ax_map.plot(sample_lon, sample_lat, "-", lw=0.95, color=sample_color, alpha=0.42, zorder=4, label=sample_label)
        ax_time.plot(sample_t - t0, sample_hgt, "-", lw=0.95, color=sample_color, alpha=0.42, zorder=4, label=sample_label)
        ax_B.semilogy(sample_t - t0, sample_B, "-", lw=0.95, color=sample_color, alpha=0.42, zorder=4, label=sample_label)
        ax_lonalt.plot(sample_lon, sample_hgt, "-", lw=0.95, color=sample_color, alpha=0.42, zorder=4, label=sample_label)
        sample_label_used = True

        sample_impact = sample.get("impact")
        if sample_impact is not None:
            sample_impact_model = sample_impact["trajectory"]
            sample_impact_order = n.argsort(sample_impact_model["times_model"])
            sample_ti = n.asarray(sample_impact_model["times_model"][sample_impact_order], dtype=float)
            sample_lati = n.asarray(sample_impact_model["lat_deg"][sample_impact_order], dtype=float)
            sample_loni = n.asarray(sample_impact_model["lon_deg"][sample_impact_order], dtype=float)
            sample_hgti = n.asarray(sample_impact_model["hgt_m"][sample_impact_order], dtype=float) / 1e3
            sample_Bi = n.asarray(sample_impact_model["B_model"][sample_impact_order], dtype=float)
            ax_map.plot(sample_loni, sample_lati, "--", lw=0.85, color=sample_color, alpha=0.34, zorder=4)
            ax_time.plot(sample_ti - t0, sample_hgti, "--", lw=0.85, color=sample_color, alpha=0.34, zorder=4)
            ax_B.semilogy(sample_ti - t0, sample_Bi, "--", lw=0.85, color=sample_color, alpha=0.34, zorder=4)
            ax_lonalt.plot(sample_loni, sample_hgti, "--", lw=0.85, color=sample_color, alpha=0.34, zorder=4)

    ax_map.plot(
        lon_model,
        lat_model,
        "-",
        lw=2.5,
        color=model_color,
        label="Best-fit trajectory",
        zorder=best_fit_zorder,
    )
    ax_map.scatter(
        lon_meas,
        lat_meas,
        s=24,
        color=measurement_color,
        linewidths=0,
        alpha=0.9,
        zorder=measurement_zorder,
        rasterized=True,
        label="Used measurements",
    )
    for i, (frag_id, lat_frag, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        ax_map.plot(
            lon_frag,
            lat_frag,
            marker="*",
            linestyle="None",
            color="gold",
            markeredgecolor="black",
            markersize=9,
            markeredgewidth=0.8,
            label="Recovered fragment" if i == 0 else None,
            zorder=16,
        )
    ax_map.set_xlabel("Longitude (deg)", fontsize=axis_label_fontsize)
    ax_map.set_ylabel("Latitude (deg)", fontsize=axis_label_fontsize)
    style_publication_axis(ax_map, tick_labelsize=tick_label_fontsize)

    ax_time.plot(
        t_model - t0,
        hgt_model / 1e3,
        "-",
        lw=2.5,
        color=model_color,
        label="Best-fit trajectory",
        zorder=best_fit_zorder,
    )
    ax_time.scatter(
        t_meas - t0,
        hgt_meas / 1e3,
        s=24,
        color=measurement_color,
        linewidths=0,
        alpha=0.9,
        zorder=measurement_zorder,
        rasterized=True,
        label="Used measurements",
    )
    ax_time.set_xlabel("Time since first measurement (s)", fontsize=axis_label_fontsize)
    ax_time.set_ylabel("Height (km)", fontsize=axis_label_fontsize)
    style_publication_axis(ax_time, tick_labelsize=tick_label_fontsize)

    ax_B.semilogy(
        t_model - t0,
        B_model,
        "-",
        lw=2.5,
        color=model_color,
        label=r"Best-fit $B(t)$",
        zorder=best_fit_zorder,
    )
    ax_B.set_xlabel("Time since first measurement (s)", fontsize=axis_label_fontsize)
    ax_B.set_ylabel(r"$B(t)$ (m$^2$ kg$^{-1}$)", fontsize=axis_label_fontsize)
    style_publication_axis(ax_B, tick_labelsize=B_tick_label_fontsize)
    ax_B.tick_params(axis="both", which="major", labelsize=B_tick_label_fontsize)
    ax_B.tick_params(axis="both", which="minor", labelsize=max(B_tick_label_fontsize - 1, 1))

    ax_lonalt.plot(
        lon_model,
        hgt_model / 1e3,
        "-",
        lw=2.5,
        color=model_color,
        label="Best-fit trajectory",
        zorder=best_fit_zorder,
    )
    ax_lonalt.scatter(
        lon_meas,
        hgt_meas / 1e3,
        s=24,
        color=measurement_color,
        linewidths=0,
        alpha=0.9,
        zorder=measurement_zorder,
        rasterized=True,
        label="Used measurements",
    )
    for i, (_, _, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        ax_lonalt.axvline(
            lon_frag,
            color="#2171b5",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            zorder=8,
            label="Recovered fragment longitude" if i == 0 else None,
        )
    ax_lonalt.set_xlabel("Longitude (deg)", fontsize=axis_label_fontsize)
    ax_lonalt.set_ylabel("Height (km)", fontsize=axis_label_fontsize)
    style_publication_axis(ax_lonalt, tick_labelsize=tick_label_fontsize)

    if impact is not None:
        impact_model = impact["trajectory"]
        impact_order = n.argsort(impact_model["times_model"])
        t_impact = impact_model["times_model"][impact_order]
        lat_impact = impact_model["lat_deg"][impact_order]
        lon_impact = impact_model["lon_deg"][impact_order]
        hgt_impact = impact_model["hgt_m"][impact_order]
        B_impact = impact_model["B_model"][impact_order]
        ax_map.plot(
            lon_impact,
            lat_impact,
            "--",
            lw=2.3,
            color=model_color,
            label="Extrapolated trajectory",
            zorder=extrapolated_zorder,
        )
        ax_map.plot(
            impact["impact_lon_deg"],
            impact["impact_lat_deg"],
            "x",
            ms=8,
            mew=2,
            label="Impact",
            color=model_color,
        )
        ax_time.plot(
            t_impact - t0,
            hgt_impact / 1e3,
            "--",
            lw=2.3,
            color=model_color,
            label="Extrapolated trajectory",
            zorder=extrapolated_zorder,
        )
        ax_time.plot(
            impact["impact_time_unix"] - t0,
            impact["impact_hgt_m"] / 1e3,
            "x",
            ms=8,
            mew=2,
            label="Impact",
            color=model_color,
        )
        ax_B.semilogy(
            t_impact - t0,
            B_impact,
            "--",
            lw=2.3,
            color=model_color,
            label=r"Extrapolated $B(t)$",
            zorder=extrapolated_zorder,
        )
        ax_lonalt.plot(
            lon_impact,
            hgt_impact / 1e3,
            "--",
            lw=2.3,
            color=model_color,
            label="Extrapolated trajectory",
            zorder=extrapolated_zorder,
        )
        ax_lonalt.plot(
            impact["impact_lon_deg"],
            impact["impact_hgt_m"] / 1e3,
            "x",
            ms=8,
            mew=2,
            label="Impact",
            color=model_color,
            zorder=10,
        )
    for ax in (ax_map, ax_B, ax_lonalt):
        ax.legend(
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.85",
            fontsize=legend_fontsize,
        )

    if "hdf5_path" in result:
        pdf_path = Path(result["hdf5_path"]).with_suffix(".pdf")
    elif "fit_ids" in result:
        pdf_path = build_fit_result_hdf5_path(result["fit_ids"]).with_suffix(".pdf")
    else:
        pdf_path = Path(__file__).with_name("ballistic_fit_publication.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    result["plot_pdf_path"] = str(pdf_path)
    print("plot_pdf_path", pdf_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf_path


def build_debug_measurement_trace(times_unix, pos_eci, label):
    times_unix = n.asarray(times_unix, dtype=float).reshape(-1)
    pos_eci = n.asarray(pos_eci, dtype=float)
    _, lon_deg, hgt_m = eci_to_geodetic(pos_eci, times_unix)
    return {
        "label": str(label),
        "times_rel_s": times_unix - float(n.min(times_unix)),
        "lon_deg": n.asarray(lon_deg, dtype=float),
        "hgt_km": n.asarray(hgt_m, dtype=float) / 1e3,
    }


def update_debug_fit_plot(debug_state, measurement_traces, model_traces, stage_name, eval_count, best_cost):
    if debug_state is None or not debug_state.get("enabled", False):
        return debug_state

    n_panels = max(len(measurement_traces), 1)
    ncols = min(3, n_panels)
    nrows = int(n.ceil(n_panels / ncols))

    fig = debug_state.get("fig")
    recreate = (
        fig is None
        or not plt.fignum_exists(fig.number)
        or debug_state.get("n_panels") != n_panels
    )
    if recreate:
        plt.ion()
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(min(14.0, 4.6 * ncols), min(8.5, 2.6 * nrows)),
            squeeze=False,
            constrained_layout=True,
        )
        debug_state["fig"] = fig
        debug_state["axes"] = axes.ravel().tolist()
        debug_state["n_panels"] = n_panels

    axes = debug_state["axes"]
    for i, ax in enumerate(axes):
        if i >= n_panels:
            ax.set_visible(False)
            continue
        ax.set_visible(True)
        ax.cla()

        meas = measurement_traces[i]
        model = model_traces[i] if i < len(model_traces) else None
        ax.plot(
            meas["lon_deg"],
            meas["hgt_km"],
            ".",
            color="0.55",
            markersize=4,
            alpha=0.9,
            label="Measurements",
        )
        if model is not None:
            ax.plot(
                model["lon_deg"],
                model["hgt_km"],
                "-",
                color="#cb181d",
                linewidth=1.6,
                label="Best fit",
            )
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title(meas["label"], fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{stage_name}: evaluation {int(eval_count)}, best cost = {float(best_cost):.6e}",
        fontsize=12,
    )
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return debug_state


def plot_density_profile(result):
    model = result["model"]
    order = n.argsort(model["times_model"])
    rho_model = model["rho_a_kg_m3"][order]
    hgt_model = model["hgt_m"][order]

    fig, ax = plt.subplots(figsize=(6, 6))
    mask_model = n.isfinite(rho_model) & (rho_model > 0.0)
    ax.semilogx(
        rho_model[mask_model],
        hgt_model[mask_model] / 1e3,
        "-",
        lw=2,
        label="Best fit model",
    )

    impact = result.get("impact")
    if impact is not None:
        impact_model = impact["trajectory"]
        impact_order = n.argsort(impact_model["times_model"])
        rho_impact = impact_model["rho_a_kg_m3"][impact_order]
        hgt_impact = impact_model["hgt_m"][impact_order]
        mask_impact = n.isfinite(rho_impact) & (rho_impact > 0.0)
        ax.semilogx(
            rho_impact[mask_impact],
            hgt_impact[mask_impact] / 1e3,
            "--",
            lw=2,
            label="Extrapolated path",
        )

    ax.set_xlabel(r"Atmospheric density (kg m$^{-3}$)")
    ax.set_ylabel("Height (km)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_specific_energy_loss_rate(result):
    plot_lon_height_colored(
        result,
        field_name="specific_energy_loss_rate_w_kg",
        colorbar_label=r"Energy loss rate per unit mass (W kg$^{-1}$)",
        cmap="inferno",
        log_color=True,
        positive_only=True,
    )


def plot_velocity_scatter(result):
    plot_lon_height_colored(
        result,
        field_name="relative_speed_m_s",
        colorbar_label=r"Velocity relative to atmosphere (m s$^{-1}$)",
        cmap="viridis",
        log_color=False,
        positive_only=False,
    )


def plot_lon_height_colored(
    result,
    field_name,
    colorbar_label,
    cmap="viridis",
    log_color=False,
    positive_only=False,
):
    model = result["model"]
    order = n.argsort(model["times_model"])
    lon_model = model["lon_deg"][order]
    hgt_model = model["hgt_m"][order] / 1e3
    values_model = model[field_name][order]

    impact = result.get("impact")
    lon_impact = n.array([], dtype=float)
    hgt_impact = n.array([], dtype=float)
    values_impact = n.array([], dtype=float)
    if impact is not None:
        impact_model = impact["trajectory"]
        impact_order = n.argsort(impact_model["times_model"])
        lon_impact = impact_model["lon_deg"][impact_order]
        hgt_impact = impact_model["hgt_m"][impact_order] / 1e3
        values_impact = impact_model[field_name][impact_order]

    mask_model = n.isfinite(values_model)
    mask_impact = n.isfinite(values_impact)
    if positive_only:
        mask_model &= values_model > 0.0
        mask_impact &= values_impact > 0.0

    valid_sets = []
    if n.any(mask_model):
        valid_sets.append(values_model[mask_model])
    if n.any(mask_impact):
        valid_sets.append(values_impact[mask_impact])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = None

    if valid_sets:
        valid_values = n.concatenate(valid_sets)
        vmin = n.nanmin(valid_values)
        vmax = n.nanmax(valid_values)
        if log_color and vmax > vmin and vmin > 0.0:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        if n.any(mask_model):
            sc = ax.scatter(
                lon_model[mask_model],
                hgt_model[mask_model],
                c=values_model[mask_model],
                s=18,
                cmap=cmap,
                norm=norm,
                linewidths=0,
                label="Best fit model",
            )
        if n.any(mask_impact):
            sc_impact = ax.scatter(
                lon_impact[mask_impact],
                hgt_impact[mask_impact],
                c=values_impact[mask_impact],
                s=16,
                cmap=cmap,
                norm=norm,
                marker="s",
                linewidths=0,
                alpha=0.9,
                label="Extrapolated path",
            )
            if sc is None:
                sc = sc_impact

        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(colorbar_label)
    else:
        ax.scatter(
            lon_model,
            hgt_model,
            s=18,
            color="0.4",
            linewidths=0,
            label="Best fit model",
        )
        if impact is not None and lon_impact.size > 0:
            ax.scatter(
                lon_impact,
                hgt_impact,
                s=16,
                color="0.6",
                marker="s",
                linewidths=0,
                label="Extrapolated path",
            )

    if impact is not None:
        ax.plot(
            lon_impact,
            hgt_impact,
            "--",
            color="tab:blue",
            linewidth=1.0,
            alpha=0.8,
            zorder=4,
        )

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Height (km)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    plt.show()



def propagate(
    p0,
    v0,
    t,
    B0=[-3, -3, -3],
    dt=0.5,
    fixed_B=None,
    start_time=None,
    stop_at_ground=False,
    density_profile=None,
    wind_model=None,
):
    """
    Simple forward propagation utility used for quick inspection/debugging.
    """
    t = n.asarray(t, dtype=float)
    if t.size == 0:
        raise ValueError("Need at least one observation time to propagate.")

    B0 = n.asarray(B0, dtype=float).reshape(-1)
    if fixed_B is None:
        mtv = ballistic_node_times(t, dt, B0.size)
        Bfun = sint.interp1d(
            mtv,
            B0,
            kind="linear",
            bounds_error=False,
            fill_value=(float(B0[0]), float(B0[-1])),
            assume_sorted=True,
        )
    else:
        B_const = fixed_B

        def Bfun(t_query):
            t_arr = n.asarray(t_query, dtype=float)
            if t_arr.ndim == 0:
                return B_const
            return n.full(t_arr.shape, B_const, dtype=float)
    p = n.asarray(p0, dtype=float).copy()
    v = n.asarray(v0, dtype=float).copy()
    if start_time is None:
        tnow = n.min(t)
    else:
        tnow = float(start_time)
    pos_eci = []
    vel_eci = []
    t_model = []
    lat_model = []
    lon_model = []
    hgt_model = []
    B_model = []
    rho_a_model = []
    specific_energy_loss_rate_model = []
    speed_model = []
    relative_speed_model = []

    pos_eci.append(p.copy())
    vel_eci.append(v.copy())
    t_model.append(tnow)
    B_now = 10**Bfun(tnow)
    lat, lon, hgt, rho_a, _, a_drag, specific_energy_loss_rate, speed, relative_speed = drag_state_eci(
        p,
        v,
        tnow,
        B_now,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    lat_model.append(lat)
    lon_model.append(lon)
    hgt_model.append(hgt)
    B_model.append(B_now)
    rho_a_model.append(rho_a)
    specific_energy_loss_rate_model.append(specific_energy_loss_rate)
    speed_model.append(speed)
    relative_speed_model.append(relative_speed)

    if stop_at_ground and hgt <= 0.0:
        stop_at_ground = False

    while tnow <= n.max(t):
        B_now = 10**Bfun(tnow)
        lat, lon, hgt, rho_a, _, a_drag, specific_energy_loss_rate, _, _ = drag_state_eci(
            p,
            v,
            tnow,
            B_now,
            density_profile=density_profile,
            wind_model=wind_model,
        )
        a_grav = gravity_accel_eci_j2(p)
        #if vmag > 0.0:
        #    a_lift0=-a_grav/n.linalg.norm(a_grav)
        #    a_lift = 0.5 * rho_a * a_lift0 * (vmag**2) * B_now * 10**C_L
        #else:
        #    a_lift = n.zeros(3, dtype=float)

        dv = (a_drag + a_grav) * dt
        v = v + dv
        p = p + v * dt
        tnow = tnow + dt
        B_now = 10**Bfun(tnow)
        lat, lon, hgt, rho_a, _, _, specific_energy_loss_rate, speed, relative_speed = drag_state_eci(
            p,
            v,
            tnow,
            B_now,
            density_profile=density_profile,
            wind_model=wind_model,
        )
        t_model.append(tnow)
        pos_eci.append(p.copy())
        vel_eci.append(v.copy())
        lat_model.append(lat)
        lon_model.append(lon)
        hgt_model.append(hgt)
        B_model.append(B_now)
        rho_a_model.append(rho_a)
        specific_energy_loss_rate_model.append(specific_energy_loss_rate)
        speed_model.append(speed)
        relative_speed_model.append(relative_speed)

        if stop_at_ground and hgt <= 0.0:
            break

    t_model = n.array(t_model)
    vel_eci = n.array(vel_eci)
    pos_eci = n.array(pos_eci)
    lat_model = n.array(lat_model)
    lon_model = n.array(lon_model)
    hgt_model = n.array(hgt_model)
    B_model = n.array(B_model)
    rho_a_model = n.array(rho_a_model)
    specific_energy_loss_rate_model = n.array(specific_energy_loss_rate_model)
    speed_model = n.array(speed_model)
    relative_speed_model = n.array(relative_speed_model)

    pos_eci_interp = sint.interp1d(
        t_model,
        pos_eci,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    vel_eci_interp = sint.interp1d(
        t_model,
        vel_eci,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    lat_interp = sint.interp1d(
        t_model,
        lat_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    lon_interp = sint.interp1d(
        t_model,
        lon_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    hgt_interp = sint.interp1d(
        t_model,
        hgt_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    B_interp = sint.interp1d(
        t_model,
        B_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    rho_a_interp = sint.interp1d(
        t_model,
        rho_a_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    specific_energy_loss_rate_interp = sint.interp1d(
        t_model,
        specific_energy_loss_rate_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    speed_interp = sint.interp1d(
        t_model,
        speed_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    relative_speed_interp = sint.interp1d(
        t_model,
        relative_speed_model,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )

    return {
        "times_model": t_model,
        "pos_eci": pos_eci,
        "vel_eci": vel_eci,
        "lat_deg": lat_model,
        "lon_deg": lon_model,
        "hgt_m": hgt_model,
        "B_model": B_model,
        "rho_a_kg_m3": rho_a_model,
        "specific_energy_loss_rate_w_kg": specific_energy_loss_rate_model,
        "speed_m_s": speed_model,
        "relative_speed_m_s": relative_speed_model,
        "pos_eci_interp": pos_eci_interp,
        "vel_eci_interp": vel_eci_interp,
        "lat_interp": lat_interp,
        "lon_interp": lon_interp,
        "hgt_interp": hgt_interp,
        "B_interp": B_interp,
        "rho_a_interp": rho_a_interp,
        "specific_energy_loss_rate_interp": specific_energy_loss_rate_interp,
        "speed_interp": speed_interp,
        "relative_speed_interp": relative_speed_interp,
        "B_node_times_unix": None if fixed_B is not None else mtv,
        "log_B_nodes": None if fixed_B is not None else B0.copy(),
        "density_profile": density_profile,
        "wind_model": wind_model,
    }


def extrapolate_best_fit_to_ground(result, max_time_ahead=3600.0, dt=None):
    """
    Continue the best-fit trajectory until it reaches the ground.

    The extrapolation starts at the last measurement time and holds the
    ballistic coefficient fixed at its fitted value at that time.
    """
    model = result["model"]
    t_start = float(n.max(result["times_unix"]))
    dt_use = float(result["dt_model"] if dt is None else dt)

    p_start = n.asarray(model["pos_eci_interp"](t_start), dtype=float)
    v_start = n.asarray(model["vel_eci_interp"](t_start), dtype=float)
    B_start = n.log10(float(model["B_interp"](t_start)))

    extrapolation = propagate(
        p_start,
        v_start,
        n.array([t_start, t_start + float(max_time_ahead)], dtype=float),
        B_start,
        dt=dt_use,
        fixed_B=B_start,
        start_time=t_start,
        stop_at_ground=True,
        density_profile=model.get("density_profile"),
        wind_model=model.get("wind_model"),
    )

    hgt_model = extrapolation["hgt_m"]
    ground_idx = n.where(hgt_model <= 0.0)[0]
    if ground_idx.size == 0:
        raise ValueError("Ground impact was not reached within max_time_ahead.")

    i1 = int(ground_idx[0])
    i0 = max(i1 - 1, 0)

    times_model = extrapolation["times_model"]
    pos_model = extrapolation["pos_eci"]
    vel_model = extrapolation["vel_eci"]
    B_model = extrapolation["B_model"]

    if i0 == i1:
        alpha = 0.0
    else:
        h0 = float(hgt_model[i0])
        h1 = float(hgt_model[i1])
        denom = h0 - h1
        alpha = 0.0 if abs(denom) < 1e-12 else n.clip(h0 / denom, 0.0, 1.0)

    impact_time_unix = times_model[i0] + alpha * (times_model[i1] - times_model[i0])
    impact_pos_eci = pos_model[i0] + alpha * (pos_model[i1] - pos_model[i0])
    impact_vel_eci = vel_model[i0] + alpha * (vel_model[i1] - vel_model[i0])
    impact_B = B_model[i0] + alpha * (B_model[i1] - B_model[i0])
    impact_specific_energy_loss_rate = (
        extrapolation["specific_energy_loss_rate_w_kg"][i0]
        + alpha
        * (
            extrapolation["specific_energy_loss_rate_w_kg"][i1]
            - extrapolation["specific_energy_loss_rate_w_kg"][i0]
        )
    )
    impact_speed = (
        extrapolation["speed_m_s"][i0]
        + alpha * (extrapolation["speed_m_s"][i1] - extrapolation["speed_m_s"][i0])
    )
    impact_relative_speed = (
        extrapolation["relative_speed_m_s"][i0]
        + alpha
        * (
            extrapolation["relative_speed_m_s"][i1]
            - extrapolation["relative_speed_m_s"][i0]
        )
    )
    impact_lat_deg, impact_lon_deg, impact_hgt_m = eci_to_geodetic(
        impact_pos_eci,
        impact_time_unix,
    )

    return {
        "impact_time_unix": float(impact_time_unix),
        "impact_lat_deg": float(impact_lat_deg),
        "impact_lon_deg": float(impact_lon_deg),
        "impact_hgt_m": float(impact_hgt_m),
        "impact_pos_eci": n.asarray(impact_pos_eci, dtype=float),
        "impact_vel_eci": n.asarray(impact_vel_eci, dtype=float),
        "impact_B": float(impact_B),
        "impact_specific_energy_loss_rate_w_kg": float(impact_specific_energy_loss_rate),
        "impact_speed_m_s": float(impact_speed),
        "impact_relative_speed_m_s": float(impact_relative_speed),
        "fixed_B": float(B_start),
        "trajectory": extrapolation,
    }


def build_model_from_parameters(x, t, dt_model, density_profile=None, wind_model=None):
    p0 = x[0:3]
    v0 = x[3:6]
    B0 = x[6:]
    model = propagate(
        p0,
        v0,
        t,
        B0=B0,
        dt=dt_model,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    return model


def residuals_lm(
    x,
    t,
    pos_eci,
    dt_model,
    gamma_fixed,
    density_profile=None,
    wind_model=None,
    residual_weights=None,
):
    bad = n.full(pos_eci.size, 1e9, dtype=float)

    if not n.all(n.isfinite(x)):
        return bad

    try:
        model = build_model_from_parameters(
            x,
            t,
            dt_model,
            density_profile=density_profile,
            wind_model=wind_model,
        )
        model_pos = n.asarray(model["pos_eci_interp"](t), dtype=float)
    except Exception:
        return bad

    if not n.all(n.isfinite(model_pos)):
        return bad

    residuals = model_pos - pos_eci
    if not n.all(n.isfinite(residuals)):
        return bad
    if residual_weights is not None:
        weights = n.asarray(residual_weights, dtype=float).reshape(-1)
        if weights.shape[0] != residuals.shape[0]:
            return bad
        residuals = residuals * n.sqrt(n.maximum(weights, 0.0))[:, None]

    return residuals.ravel()


def objective_fmin(
    x,
    t,
    pos_eci,
    dt_model,
    gamma_fixed,
    density_profile=None,
    wind_model=None,
    debug_state=None,
    debug_measurements=None,
    stage_name="fmin",
):
    residuals = residuals_lm(
        x,
        t,
        pos_eci,
        dt_model,
        gamma_fixed,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    s = float(n.sum(residuals**2))
    if debug_state is not None:
        debug_state["eval_count"] = int(debug_state.get("eval_count", 0)) + 1
        if n.isfinite(s) and s < float(debug_state.get("best_cost", n.inf)):
            debug_state["best_cost"] = float(s)
            debug_state["best_x"] = n.asarray(x, dtype=float).copy()

        every = max(int(debug_state.get("every", 100)), 1)
        if debug_state.get("enabled", False) and debug_state["eval_count"] % every == 0:
            best_x = debug_state.get("best_x")
            if best_x is not None:
                try:
                    best_model = build_model_from_parameters(
                        best_x,
                        t,
                        dt_model,
                        density_profile=density_profile,
                        wind_model=wind_model,
                    )
                    update_debug_fit_plot(
                        debug_state,
                        debug_measurements or [],
                        [
                            {
                                "label": debug_measurements[0]["label"] if debug_measurements else "fit",
                                "lon_deg": n.asarray(best_model["lon_deg"], dtype=float),
                                "hgt_km": n.asarray(best_model["hgt_m"], dtype=float) / 1e3,
                            }
                        ],
                        stage_name,
                        debug_state["eval_count"],
                        debug_state["best_cost"],
                    )
                except Exception:
                    pass
    # don't remove this
    print(s)
    return s


def parameter_step_sizes(x):
    x = n.asarray(x, dtype=float)
    steps = n.full(x.size, 1e-3, dtype=float)
    if x.size >= 3:
        steps[0:3] = 10.0
    if x.size >= 6:
        steps[3:6] = 0.1
    return steps


def finite_difference_jacobian(fun, x0, steps):
    x0 = n.asarray(x0, dtype=float)
    steps = n.asarray(steps, dtype=float)
    y0 = n.atleast_1d(n.asarray(fun(x0), dtype=float))
    jac = n.empty((y0.size, x0.size), dtype=float)

    for i in range(x0.size):
        step = float(steps[i])
        if not n.isfinite(step) or step <= 0.0:
            step = 1e-6
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += step
        xm[i] -= step
        yp = n.atleast_1d(n.asarray(fun(xp), dtype=float))
        ym = n.atleast_1d(n.asarray(fun(xm), dtype=float))
        jac[:, i] = (yp - ym) / (2.0 * step)

    return jac


def estimate_parameter_covariance(jacobian, residual_vector):
    jacobian = n.asarray(jacobian, dtype=float)
    residual_vector = n.asarray(residual_vector, dtype=float).reshape(-1)

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2D array")

    m, n_params = jacobian.shape
    if m <= n_params:
        return None

    dof = m - n_params
    rss = float(n.sum(residual_vector**2))
    sigma2 = rss / dof
    covariance = sigma2 * n.linalg.pinv(jacobian.T @ jacobian)

    return {
        "covariance": covariance,
        "std": n.sqrt(n.maximum(n.diag(covariance), 0.0)),
        "rss": rss,
        "dof": int(dof),
        "sigma2": float(sigma2),
    }


def run_fmin_fit_stage(
    x0,
    t,
    pos_eci,
    dt_model,
    gamma_fixed,
    density_profile=None,
    wind_model=None,
    stage_name="fmin",
    debug_plot=False,
    debug_plot_every=100,
    verbose=0,
):
    debug_state = {
        "enabled": bool(debug_plot),
        "every": int(debug_plot_every),
        "eval_count": 0,
        "best_cost": n.inf,
        "best_x": None,
        "fig": None,
        "axes": None,
        "n_panels": None,
    }
    debug_measurements = [
        build_debug_measurement_trace(
            t,
            pos_eci,
            stage_name,
        )
    ]
    xopt, fopt, iterations, funcalls, warnflag = so.fmin(
        objective_fmin,
        x0,
        args=(
            t,
            pos_eci,
            dt_model,
            gamma_fixed,
            density_profile,
            wind_model,
            debug_state,
            debug_measurements,
            stage_name,
        ),
        full_output=True,
        maxiter=8000,
        maxfun=10000,
        disp=bool(verbose > 1),
    )

    if debug_state.get("enabled", False):
        try:
            final_model = build_model_from_parameters(
                xopt,
                t,
                dt_model,
                density_profile=density_profile,
                wind_model=wind_model,
            )
            update_debug_fit_plot(
                debug_state,
                debug_measurements,
                [
                    {
                        "label": debug_measurements[0]["label"],
                        "lon_deg": n.asarray(final_model["lon_deg"], dtype=float),
                        "hgt_km": n.asarray(final_model["hgt_m"], dtype=float) / 1e3,
                    }
                ],
                stage_name,
                max(int(debug_state.get("eval_count", 0)), int(funcalls)),
                min(float(debug_state.get("best_cost", n.inf)), float(fopt)),
            )
        except Exception:
            pass

    residual_vector = residuals_lm(
        xopt,
        t,
        pos_eci,
        dt_model,
        gamma_fixed,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    residual_jac = finite_difference_jacobian(
        lambda x: residuals_lm(
            x,
            t,
            pos_eci,
            dt_model,
            gamma_fixed,
            density_profile=density_profile,
            wind_model=wind_model,
        ),
        xopt,
        parameter_step_sizes(xopt),
    )
    covariance_info = estimate_parameter_covariance(residual_jac, residual_vector)
    model = build_model_from_parameters(
        xopt,
        t,
        dt_model,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    optimizer = {
        "method": "fmin",
        "stage": str(stage_name),
        "x": xopt,
        "fun": float(fopt),
        "nit": int(iterations),
        "nfev": int(funcalls),
        "warnflag": int(warnflag),
        "success": bool(warnflag == 0),
    }
    return {
        "xopt": n.asarray(xopt, dtype=float).copy(),
        "optimizer": optimizer,
        "covariance_info": covariance_info,
        "model": model,
        "residual_vector": residual_vector,
    }


def joint_shared_start_parameter_step_sizes(
    n_groups,
    n_paths,
    n_ballistic_nodes,
    fixed_start_positions=None,
):
    group_width = 4 if fixed_start_positions is not None else 7
    local_width = max(int(n_ballistic_nodes) - 1, 0)  # path-specific remaining log_B nodes
    steps = n.full(group_width * n_groups + local_width * n_paths, 1e-3, dtype=float)
    for i in range(n_groups):
        base = group_width * i
        if fixed_start_positions is None:
            steps[base:base + 3] = 10.0
            steps[base + 3:base + 6] = 0.1
            steps[base + 6] = 1e-3
        else:
            steps[base:base + 3] = 0.1
            steps[base + 3] = 1e-3
    return steps


def joint_shared_start_parameter_bounds(
    n_groups,
    n_paths,
    n_ballistic_nodes,
    fixed_start_positions=None,
    log_B_bounds=(-6.0, -1.5),
):
    group_width = 4 if fixed_start_positions is not None else 7
    local_width = max(int(n_ballistic_nodes) - 1, 0)
    lower = n.full(group_width * n_groups + local_width * n_paths, -n.inf, dtype=float)
    upper = n.full(group_width * n_groups + local_width * n_paths, n.inf, dtype=float)
    log_B_lower, log_B_upper = (float(log_B_bounds[0]), float(log_B_bounds[1]))

    for i in range(n_groups):
        base = group_width * i
        log_B_index = base + (3 if fixed_start_positions is not None else 6)
        lower[log_B_index] = log_B_lower
        upper[log_B_index] = log_B_upper

    local_base0 = group_width * n_groups
    for path_index in range(n_paths):
        local_base = local_base0 + local_width * path_index
        lower[local_base:local_base + local_width] = log_B_lower
        upper[local_base:local_base + local_width] = log_B_upper

    return lower, upper


def joint_shared_start_group_order(prepared_paths):
    group_order = []
    for path in prepared_paths:
        shared_start_id = path["shared_start_id"]
        if shared_start_id not in group_order:
            group_order.append(shared_start_id)
    return tuple(group_order)


def joint_shared_start_decode_local_x(
    x,
    path_index,
    prepared_path,
    group_index,
    n_groups,
    n_ballistic_nodes,
    fixed_start_positions=None,
):
    x = n.asarray(x, dtype=float)
    group_width = 4 if fixed_start_positions is not None else 7
    group_base = group_width * group_index[prepared_path["shared_start_id"]]
    local_width = max(int(n_ballistic_nodes) - 1, 0)
    local_base = group_width * n_groups + local_width * path_index
    if fixed_start_positions is None:
        shared_state = x[group_base:group_base + 6]
        shared_log_B0 = x[group_base + 6]
    else:
        shared_start_id = prepared_path["shared_start_id"]
        p0_fixed = n.asarray(fixed_start_positions[shared_start_id], dtype=float).reshape(3)
        v0_shared = x[group_base:group_base + 3]
        shared_state = n.concatenate((p0_fixed, v0_shared))
        shared_log_B0 = x[group_base + 3]
    return n.concatenate(
        (
            shared_state,
            n.concatenate(
                (
                    n.array([shared_log_B0], dtype=float),
                    x[local_base:local_base + local_width],
                )
            ),
        )
    )


def joint_shared_start_extract_local_covariance(
    covariance,
    path_index,
    prepared_path,
    group_index,
    n_groups,
    n_ballistic_nodes,
    fixed_start_positions=None,
):
    if covariance is None:
        return None
    group_width = 4 if fixed_start_positions is not None else 7
    group_base = group_width * group_index[prepared_path["shared_start_id"]]
    local_width = max(int(n_ballistic_nodes) - 1, 0)
    local_base = group_width * n_groups + local_width * path_index
    if fixed_start_positions is None:
        idx = list(range(group_base, group_base + 7)) + list(range(local_base, local_base + local_width))
        return covariance[n.ix_(idx, idx)]

    idx = list(range(group_base, group_base + 4)) + list(range(local_base, local_base + local_width))
    variable_covariance = covariance[n.ix_(idx, idx)]
    full_width = 6 + int(n_ballistic_nodes)
    full_covariance = n.zeros((full_width, full_width), dtype=float)
    variable_local_idx = list(range(3, 7 + local_width))
    full_covariance[n.ix_(variable_local_idx, variable_local_idx)] = variable_covariance
    return full_covariance


def run_joint_shared_start_fmin_stage(
    prepared_paths,
    shared_start_order,
    x0,
    n_ballistic_nodes,
    dt_model,
    gamma_fixed,
    wind_models=None,
    stage_name="joint_fmin",
    debug_plot=False,
    debug_plot_every=100,
    verbose=0,
    fixed_start_positions=None,
    skip_fmin=False,
):
    n_paths = len(prepared_paths)
    n_groups = len(shared_start_order)
    if wind_models is None:
        wind_models = [None] * n_paths
    if len(wind_models) != n_paths:
        raise ValueError("wind_models must match the number of prepared paths")

    group_index = {fid: i for i, fid in enumerate(shared_start_order)}
    debug_state = {
        "enabled": bool(debug_plot),
        "every": int(debug_plot_every),
        "eval_count": 0,
        "best_cost": n.inf,
        "best_x": None,
        "fig": None,
        "axes": None,
        "n_panels": None,
    }
    debug_measurements = [
        build_debug_measurement_trace(
            path["times_unix"],
            path["pos_eci"],
            " -> ".join(path["fit_ids"]),
        )
        for path in prepared_paths
    ]

    def residuals_joint(x):
        residuals = []
        for i, (path, wind_model) in enumerate(zip(prepared_paths, wind_models)):
            local_x = joint_shared_start_decode_local_x(
                x,
                i,
                path,
                group_index,
                n_groups,
                n_ballistic_nodes,
                fixed_start_positions=fixed_start_positions,
            )
            residuals.append(
                residuals_lm(
                    local_x,
                    path["times_unix"],
                    path["pos_eci"],
                    dt_model,
                    gamma_fixed,
                    density_profile=path["density_profile"],
                    wind_model=wind_model,
                    residual_weights=path.get("residual_weights"),
                )
            )
        return n.concatenate(residuals)

    def objective_joint(x):
        residual_vector = residuals_joint(x)
        cost = float(n.sum(residual_vector**2))
        debug_state["eval_count"] = int(debug_state.get("eval_count", 0)) + 1
        if n.isfinite(cost) and cost < float(debug_state.get("best_cost", n.inf)):
            debug_state["best_cost"] = float(cost)
            debug_state["best_x"] = n.asarray(x, dtype=float).copy()
        every = max(int(debug_state.get("every", 100)), 1)
        if debug_state.get("enabled", False) and debug_state["eval_count"] % every == 0:
            best_x = debug_state.get("best_x")
            if best_x is not None:
                try:
                    best_model_traces = []
                    for i, (path, wind_model) in enumerate(zip(prepared_paths, wind_models)):
                        local_best_x = joint_shared_start_decode_local_x(
                            best_x,
                            i,
                            path,
                            group_index,
                            n_groups,
                            n_ballistic_nodes,
                            fixed_start_positions=fixed_start_positions,
                        )
                        best_model = build_model_from_parameters(
                            local_best_x,
                            path["times_unix"],
                            dt_model,
                            density_profile=path["density_profile"],
                            wind_model=wind_model,
                        )
                        best_model_traces.append(
                            {
                                "label": " -> ".join(path["fit_ids"]),
                                "lon_deg": n.asarray(best_model["lon_deg"], dtype=float),
                                "hgt_km": n.asarray(best_model["hgt_m"], dtype=float) / 1e3,
                            }
                        )
                    update_debug_fit_plot(
                        debug_state,
                        debug_measurements,
                        best_model_traces,
                        stage_name,
                        debug_state["eval_count"],
                        debug_state["best_cost"],
                    )
                except Exception:
                    pass
        if verbose > 0:
            print(cost)
        return cost

    if skip_fmin:
        if verbose > 0:
            print("skipping fmin stage; refining existing fit with bounded least_squares")
        xopt = n.asarray(x0, dtype=float).copy()
        fopt = float(n.sum(residuals_joint(xopt) ** 2))
        iterations = 0
        funcalls = 1
        warnflag = 0
        method = "least_squares_joint"
    else:
        xopt, fopt, iterations, funcalls, warnflag = so.fmin(
            objective_joint,
            n.asarray(x0, dtype=float),
            full_output=True,
            maxiter=max(8000, 3000 * n_paths),
            maxfun=max(10000, 4000 * n_paths),
            disp=bool(verbose > 1),
        )
        method = "fmin_joint"
    ls_nfev = 0
    ls_success = False
    ls_message = ""
    try:
        lower_bounds, upper_bounds = joint_shared_start_parameter_bounds(
            n_groups,
            n_paths,
            n_ballistic_nodes,
            fixed_start_positions=fixed_start_positions,
        )
        xopt = n.minimum(n.maximum(n.asarray(xopt, dtype=float), lower_bounds), upper_bounds)
        x_scale = n.maximum(n.abs(xopt), 1.0)

        def residuals_joint_least_squares(x):
            residual_vector = residuals_joint(x)
            cost = float(n.sum(residual_vector**2))
            debug_state["eval_count"] = int(debug_state.get("eval_count", 0)) + 1
            if n.isfinite(cost) and cost < float(debug_state.get("best_cost", n.inf)):
                debug_state["best_cost"] = float(cost)
                debug_state["best_x"] = n.asarray(x, dtype=float).copy()
            every = max(int(debug_state.get("every", 100)), 1)
            if debug_state.get("enabled", False) and debug_state["eval_count"] % every == 0:
                try:
                    model_traces = []
                    for i, (path, wind_model) in enumerate(zip(prepared_paths, wind_models)):
                        local_x = joint_shared_start_decode_local_x(
                            x,
                            i,
                            path,
                            group_index,
                            n_groups,
                            n_ballistic_nodes,
                            fixed_start_positions=fixed_start_positions,
                        )
                        model = build_model_from_parameters(
                            local_x,
                            path["times_unix"],
                            dt_model,
                            density_profile=path["density_profile"],
                            wind_model=wind_model,
                        )
                        model_traces.append(
                            {
                                "label": " -> ".join(path["fit_ids"]),
                                "lon_deg": n.asarray(model["lon_deg"], dtype=float),
                                "hgt_km": n.asarray(model["hgt_m"], dtype=float) / 1e3,
                            }
                        )
                    update_debug_fit_plot(
                        debug_state,
                        debug_measurements,
                        model_traces,
                        "%s least_squares" % stage_name,
                        debug_state["eval_count"],
                        cost,
                    )
                except Exception:
                    pass
            return residual_vector

        ls_result = so.least_squares(
            residuals_joint_least_squares,
            xopt,
            bounds=(lower_bounds, upper_bounds),
            x_scale=x_scale,
            loss="soft_l1",
            f_scale=1000.0,
            max_nfev=max(2000, 700 * n.asarray(xopt).size),
            verbose=0,
        )
        ls_nfev = int(ls_result.nfev)
        ls_success = bool(ls_result.success)
        ls_message = str(ls_result.message)
        ls_cost_sum = float(n.sum(residuals_joint(ls_result.x) ** 2))
        if n.isfinite(ls_cost_sum) and ls_cost_sum <= float(n.sum(residuals_joint(xopt) ** 2)):
            xopt = n.asarray(ls_result.x, dtype=float)
            fopt = ls_cost_sum
            warnflag = 0 if ls_success else warnflag
            method = "fmin_then_least_squares_joint"
    except Exception as exc:
        ls_message = "least_squares refinement failed: %s" % (exc,)

    if debug_state.get("enabled", False):
        try:
            final_model_traces = []
            for i, (path, wind_model) in enumerate(zip(prepared_paths, wind_models)):
                local_x = joint_shared_start_decode_local_x(
                    xopt,
                    i,
                    path,
                    group_index,
                    n_groups,
                    n_ballistic_nodes,
                    fixed_start_positions=fixed_start_positions,
                )
                final_model = build_model_from_parameters(
                    local_x,
                    path["times_unix"],
                    dt_model,
                    density_profile=path["density_profile"],
                    wind_model=wind_model,
                )
                final_model_traces.append(
                    {
                        "label": " -> ".join(path["fit_ids"]),
                        "lon_deg": n.asarray(final_model["lon_deg"], dtype=float),
                        "hgt_km": n.asarray(final_model["hgt_m"], dtype=float) / 1e3,
                    }
                )
            update_debug_fit_plot(
                debug_state,
                debug_measurements,
                final_model_traces,
                stage_name,
                max(int(debug_state.get("eval_count", 0)), int(funcalls)),
                min(float(debug_state.get("best_cost", n.inf)), float(fopt)),
            )
        except Exception:
            pass

    residual_vector = residuals_joint(xopt)
    residual_jac = finite_difference_jacobian(
        residuals_joint,
        xopt,
        joint_shared_start_parameter_step_sizes(
            n_groups,
            n_paths,
            n_ballistic_nodes,
            fixed_start_positions=fixed_start_positions,
        ),
    )
    covariance_info = estimate_parameter_covariance(residual_jac, residual_vector)

    models = []
    local_xs = []
    for i, (path, wind_model) in enumerate(zip(prepared_paths, wind_models)):
        local_x = joint_shared_start_decode_local_x(
            xopt,
            i,
            path,
            group_index,
            n_groups,
            n_ballistic_nodes,
            fixed_start_positions=fixed_start_positions,
        )
        local_xs.append(local_x)
        models.append(
            build_model_from_parameters(
                local_x,
                path["times_unix"],
                dt_model,
                density_profile=path["density_profile"],
                wind_model=wind_model,
            )
        )

    optimizer = {
        "method": method,
        "stage": str(stage_name),
        "x": n.asarray(xopt, dtype=float).copy(),
        "fun": float(fopt),
        "nit": int(iterations),
        "nfev": int(funcalls) + int(ls_nfev),
        "warnflag": int(warnflag),
        "success": bool(warnflag == 0),
        "least_squares_nfev": int(ls_nfev),
        "least_squares_success": bool(ls_success),
        "least_squares_message": str(ls_message),
    }
    return {
        "xopt": n.asarray(xopt, dtype=float).copy(),
        "optimizer": optimizer,
        "covariance_info": covariance_info,
        "residual_vector": residual_vector,
        "models": models,
        "local_xs": local_xs,
        "group_index": group_index,
        "shared_start_order": tuple(shared_start_order),
    }


def build_path_result_from_stage(
    prepared_path,
    local_x,
    local_covariance_info,
    model,
    optimizer,
    dt_model,
    wind_model_info,
    optimizer_zero_wind=None,
    optimizer_era5=None,
    plot_context_fragment_geo_pos=None,
    filename_prefix="ballistic_fit_sharedstart",
    verbose=0,
):
    local_x = n.asarray(local_x, dtype=float)
    p0_hat_eci = local_x[0:3]
    v0_hat_eci = local_x[3:6]
    B0 = local_x[6:]

    result = {
        "times_unix": prepared_path["times_unix"],
        "pos_ecef": prepared_path["pos_ecef"],
        "pos_ecef_err": prepared_path["pos_ecef_err"],
        "pos_eci": prepared_path["pos_eci"],
        "fit_ids": prepared_path["fit_ids"],
        "shared_start_id": prepared_path["shared_start_id"],
        "joint_fit": True,
        "p0_guess_ecef": prepared_path["p0_guess_ecef"],
        "v0_guess_ecef": prepared_path["v0_guess_ecef"],
        "p0_guess_eci": prepared_path["p0_guess_eci"],
        "v0_guess_eci": prepared_path["v0_guess_eci"],
        "p0_hat_eci": p0_hat_eci,
        "v0_hat_eci": v0_hat_eci,
        "B0_hat": B0,
        "B_node_times_unix": model.get("B_node_times_unix"),
        "fit_stage": optimizer["stage"],
        "dt_model": dt_model,
        "optimizer": dict(optimizer),
        "optimizer_zero_wind": None if optimizer_zero_wind is None else dict(optimizer_zero_wind),
        "optimizer_era5": None if optimizer_era5 is None else dict(optimizer_era5),
        "parameter_covariance": None if local_covariance_info is None else local_covariance_info["covariance"],
        "parameter_std": None if local_covariance_info is None else local_covariance_info["std"],
        "density_profile_altitude_m": prepared_path["density_profile"]["altitude_grid_m"],
        "density_profile_rho_kg_m3": prepared_path["density_profile"]["rho_grid_kg_m3"],
        "density_profile_reference_time_unix": prepared_path["density_profile"]["reference_time_unix"],
        "density_profile_reference_lat_deg": prepared_path["density_profile"]["reference_lat_deg"],
        "density_profile_reference_lon_deg": prepared_path["density_profile"]["reference_lon_deg"],
        "wind_model_info": wind_model_info,
        "specific_energy_loss_rate_w_kg": model["specific_energy_loss_rate_w_kg"],
        "specific_energy_loss_rate_interp": model["specific_energy_loss_rate_interp"],
        "speed_m_s": model["speed_m_s"],
        "speed_interp": model["speed_interp"],
        "relative_speed_m_s": model["relative_speed_m_s"],
        "relative_speed_interp": model["relative_speed_interp"],
        "fit_parameter_names": tuple(
            [
                "p0_eci_x", "p0_eci_y", "p0_eci_z",
                "v0_eci_x", "v0_eci_y", "v0_eci_z",
            ]
            + [f"log_B{i:02d}" for i in range(B0.size)]
        ),
        "model": model,
    }

    try:
        result["impact"] = extrapolate_best_fit_to_ground(result)
    except ValueError as exc:
        result["impact"] = None
        result["impact_error"] = str(exc)
    else:
        try:
            result["impact_uncertainty"] = estimate_impact_uncertainty(
                local_x,
                local_covariance_info,
                prepared_path["times_unix"],
                dt_model,
                density_profile=prepared_path["density_profile"],
                wind_model=model.get("wind_model"),
            )
        except ValueError as exc:
            result["impact_uncertainty"] = None
            result["impact_uncertainty_error"] = str(exc)
        except Exception as exc:
            result["impact_uncertainty"] = None
            result["impact_uncertainty_error"] = str(exc)

    if "impact_uncertainty" not in result:
        result["impact_uncertainty"] = None

    hdf5_path = save_result_to_hdf5(
        result,
        prepared_path["fit_ids"],
        filename_prefix=filename_prefix,
    )
    result["hdf5_path"] = str(hdf5_path)
    if plot_context_fragment_geo_pos is not None:
        result["plot_context_fragment_geo_pos"] = plot_context_fragment_geo_pos

    if verbose > 0:
        print("shared_start_id", prepared_path["shared_start_id"])
        print("fit_ids", " -> ".join(prepared_path["fit_ids"]))
        print("cost", optimizer["fun"])
        print("success", optimizer["success"])
        print("hdf5_path", result["hdf5_path"])
        if result["impact"] is not None:
            print("impact_lat_deg", result["impact"]["impact_lat_deg"])
            print("impact_lon_deg", result["impact"]["impact_lon_deg"])
            print("impact_time_unix", result["impact"]["impact_time_unix"])
        plot_ballistic_fit(result, show=bool(verbose > 1))

    return result


def fit_multiple_paths_shared_start_ballistic_coefficient(
    path_specs,
    gamma=0.5,
    B0_guess=[-3, -3, -3],
    plot_context_fragment_geo_pos=None,
    use_initial_hdf5=True,
    initial_filename_prefix="ballistic_fit_sharedstart",
    debug_plot=False,
    debug_plot_every=100,
    verbose=2,
    fixed_start_position=True,
    refine_existing_hdf5_only=False,
):
    if len(path_specs) == 0:
        raise ValueError("path_specs must contain at least one path")

    B0_guess = n.asarray(B0_guess, dtype=float).reshape(-1)
    if B0_guess.size < 2:
        raise ValueError("B0_guess must contain at least two node values.")
    n_ballistic_nodes = int(B0_guess.size)

    gamma_fixed = float(gamma)
    dt_model = 2.0

    prepared_paths = []
    initial_sources = []
    for spec in path_specs:
        prepared_paths.append(
            prepare_fragment_fit_data(
                spec["fragment_pos"],
                spec["fragment_pos_err"],
                spec["fragment_times"],
                spec["fit_ids"],
                terminal_weight=spec.get("terminal_weight", 1.0),
                terminal_weight_seconds=spec.get("terminal_weight_seconds", 0.0),
            )
        )
        initial_hdf5_path = None
        if use_initial_hdf5:
            initial_hdf5_path = spec.get("initial_hdf5_path")
            if initial_hdf5_path is None:
                candidate_path = build_fit_result_hdf5_path(
                    spec["fit_ids"],
                    filename_prefix=initial_filename_prefix,
                )
                if candidate_path.exists():
                    initial_hdf5_path = candidate_path

        initial_source = None
        if initial_hdf5_path is not None and Path(initial_hdf5_path).exists():
            if verbose > 0:
                print(
                    "restarting shared fit path %s from %s"
                    % (" -> ".join(str(fid) for fid in spec["fit_ids"]), Path(initial_hdf5_path).name)
                )
            initial_source = load_fit_initial_guess_from_hdf5(initial_hdf5_path)
        else:
            initial_source = spec.get("initial_result")
        initial_sources.append(initial_source)

    shared_start_order = joint_shared_start_group_order(prepared_paths)
    invalid_shared_start_ids = [
        shared_start_id
        for shared_start_id in shared_start_order
        if str(shared_start_id) not in ("1", "2")
    ]
    if len(invalid_shared_start_ids) > 0:
        raise ValueError(
            "Shared-start fit only supports root fragment families '1' and '2'; got %s"
            % (invalid_shared_start_ids,)
        )
    group_initial_p0 = {shared_start_id: [] for shared_start_id in shared_start_order}
    group_initial_v0 = {shared_start_id: [] for shared_start_id in shared_start_order}
    group_initial_log_B0 = {shared_start_id: [] for shared_start_id in shared_start_order}
    group_paths = {shared_start_id: [] for shared_start_id in shared_start_order}
    fixed_start_positions = None
    if fixed_start_position:
        fixed_start_positions = {}
        for shared_start_id in shared_start_order:
            root_path = None
            for prepared_path in prepared_paths:
                if (
                    prepared_path["shared_start_id"] == shared_start_id
                    and tuple(prepared_path["fit_ids"]) == (str(shared_start_id),)
                ):
                    root_path = prepared_path
                    break
            if root_path is None:
                root_path = next(
                    path for path in prepared_paths if path["shared_start_id"] == shared_start_id
                )
            fixed_start_positions[shared_start_id] = n.asarray(
                root_path["p0_guess_eci"],
                dtype=float,
            ).reshape(3)
    x0_parts = []

    local_initials = []
    initial_source_used = []
    for prepared_path, initial_source in zip(prepared_paths, initial_sources):
        used_initial_source = False
        if initial_source is not None:
            p0_init = n.asarray(initial_source["p0_hat_eci"], dtype=float).reshape(3)
            v0_init = n.asarray(initial_source["v0_hat_eci"], dtype=float).reshape(3)
            B_init = n.asarray(initial_source["B0_hat"], dtype=float).reshape(-1)
            if B_init.size != n_ballistic_nodes:
                if verbose > 0:
                    print(
                        "ignoring incompatible restart for %s: expected %d B0_hat values, got %d"
                        % (
                            " -> ".join(str(fid) for fid in prepared_path["fit_ids"]),
                            n_ballistic_nodes,
                            B_init.size,
                        )
                    )
                p0_init = prepared_path["p0_guess_eci"]
                v0_init = prepared_path["v0_guess_eci"]
                B_init = B0_guess.copy()
            else:
                used_initial_source = True
        else:
            p0_init = prepared_path["p0_guess_eci"]
            v0_init = prepared_path["v0_guess_eci"]
            B_init = B0_guess.copy()
        initial_source_used.append(used_initial_source)

        group_initial_p0[prepared_path["shared_start_id"]].append(n.asarray(p0_init, dtype=float).copy())
        group_initial_v0[prepared_path["shared_start_id"]].append(n.asarray(v0_init, dtype=float).copy())
        group_initial_log_B0[prepared_path["shared_start_id"]].append(float(B_init[0]))
        group_paths[prepared_path["shared_start_id"]].append(tuple(prepared_path["fit_ids"]))
        local_initials.append(
            {
                "log_B_rest": n.asarray(B_init[1:], dtype=float).copy(),
            }
        )

    for shared_start_id in shared_start_order:
        p0_values = group_initial_p0[shared_start_id]
        v0_values = group_initial_v0[shared_start_id]
        log_B0_values = group_initial_log_B0[shared_start_id]
        if len(p0_values) == 0 or len(v0_values) == 0 or len(log_B0_values) == 0:
            exemplar_path = next(
                path for path in prepared_paths if path["shared_start_id"] == shared_start_id
            )
            x0_parts.extend(n.asarray(exemplar_path["v0_guess_eci"], dtype=float).tolist())
            x0_parts.append(float(B0_guess[0]))
        else:
            x0_parts.extend(n.mean(n.vstack(v0_values), axis=0).tolist())
            x0_parts.append(float(n.mean(log_B0_values)))
        if not fixed_start_position:
            if len(p0_values) == 0:
                p0_init = n.asarray(exemplar_path["p0_guess_eci"], dtype=float)
            else:
                p0_init = n.mean(n.vstack(p0_values), axis=0)
            x0_parts[-4:-4] = p0_init.tolist()

    for local_init in local_initials:
        x0_parts.extend(local_init["log_B_rest"].tolist())

    x0 = n.asarray(x0_parts, dtype=float)
    skip_fmin = bool(refine_existing_hdf5_only) and all(initial_source_used)
    if refine_existing_hdf5_only and not skip_fmin and verbose > 0:
        print("refine_existing_hdf5_only requested, but not all restart files are compatible; running fmin")

    if verbose > 0:
        print("shared start state and ballistic coefficient groups:")
        for shared_start_id in shared_start_order:
            members = group_paths[shared_start_id]
            print("  start fragment %s:" % (shared_start_id))
            if fixed_start_positions is not None:
                print(
                    "    fixed p0 at first measurement ECI %s"
                    % n.array2string(fixed_start_positions[shared_start_id], precision=3)
                )
            for fit_ids in members:
                print("    shares p0, v0, and B0 with path %s" % (" -> ".join(fit_ids)))

    zero_wind_stage = run_joint_shared_start_fmin_stage(
        prepared_paths,
        shared_start_order,
        x0,
        n_ballistic_nodes,
        dt_model,
        gamma_fixed,
        wind_models=None,
        stage_name="zero_wind_joint",
        debug_plot=debug_plot,
        debug_plot_every=debug_plot_every,
        verbose=verbose,
        fixed_start_positions=fixed_start_positions,
        skip_fmin=skip_fmin,
    )

    zero_wind_models = zero_wind_stage["models"]
    zero_wind_wind_models = []
    zero_wind_wind_model_info = []
    for prepared_path, model in zip(prepared_paths, zero_wind_models):
        zero_wind_result = {
            "model": model,
            "times_unix": prepared_path["times_unix"],
            "dt_model": dt_model,
        }

        wind_model = None
        wind_model_info = {
            "type": "corotating_fallback",
            "reason": "ERA5 refinement not attempted",
        }
        try:
            wind_model, wind_model_info = build_cached_era5_wind_model_from_result(
                zero_wind_result,
                max_time_ahead=3600.0,
                verbose=verbose,
            )
        except Exception as exc:
            wind_model = None
            wind_model_info = {
                "type": "corotating_fallback",
                "error": str(exc),
            }

        zero_wind_wind_models.append(wind_model)
        zero_wind_wind_model_info.append(wind_model_info)

    era5_stage = None
    final_stage = zero_wind_stage
    final_wind_model_info = list(zero_wind_wind_model_info)

    if any(wind_model is not None for wind_model in zero_wind_wind_models):
        if verbose > 0:
            print("rerunning joint fit with cached ERA5 wind profiles")
        era5_stage = run_joint_shared_start_fmin_stage(
            prepared_paths,
            shared_start_order,
            zero_wind_stage["xopt"],
            n_ballistic_nodes,
            dt_model,
            gamma_fixed,
            wind_models=zero_wind_wind_models,
            stage_name="era5_profile_joint",
            debug_plot=debug_plot,
            debug_plot_every=debug_plot_every,
            verbose=verbose,
            fixed_start_positions=fixed_start_positions,
            skip_fmin=skip_fmin,
        )
        final_stage = era5_stage
        final_wind_model_info = list(zero_wind_wind_model_info)

    final_covariance = None if final_stage["covariance_info"] is None else final_stage["covariance_info"]["covariance"]
    path_results = []
    for i, prepared_path in enumerate(prepared_paths):
        local_covariance_info = None
        if final_stage["covariance_info"] is not None:
            local_covariance = joint_shared_start_extract_local_covariance(
                final_covariance,
                i,
                prepared_path,
                final_stage["group_index"],
                len(final_stage["shared_start_order"]),
                n_ballistic_nodes,
                fixed_start_positions=fixed_start_positions,
            )
            local_covariance_info = {
                "covariance": local_covariance,
                "std": n.sqrt(n.maximum(n.diag(local_covariance), 0.0)),
                "rss": final_stage["covariance_info"]["rss"],
                "dof": final_stage["covariance_info"]["dof"],
                "sigma2": final_stage["covariance_info"]["sigma2"],
            }

        path_results.append(
            build_path_result_from_stage(
                prepared_path,
                final_stage["local_xs"][i],
                local_covariance_info,
                final_stage["models"][i],
                final_stage["optimizer"],
                optimizer_zero_wind=zero_wind_stage["optimizer"],
                optimizer_era5=None if era5_stage is None else era5_stage["optimizer"],
                dt_model=dt_model,
                wind_model_info=final_wind_model_info[i],
                plot_context_fragment_geo_pos=plot_context_fragment_geo_pos,
                verbose=verbose,
            )
        )

    shared_start_p0_eci = {}
    shared_start_v0_eci = {}
    shared_start_log_B0 = {}
    for shared_start_id in final_stage["shared_start_order"]:
        group_width = 4 if fixed_start_positions is not None else 7
        group_base = group_width * final_stage["group_index"][shared_start_id]
        if fixed_start_positions is None:
            shared_start_p0_eci[shared_start_id] = n.asarray(
                final_stage["xopt"][group_base:group_base + 3],
                dtype=float,
            ).copy()
            shared_start_v0_eci[shared_start_id] = n.asarray(
                final_stage["xopt"][group_base + 3:group_base + 6],
                dtype=float,
            ).copy()
            shared_start_log_B0[shared_start_id] = float(final_stage["xopt"][group_base + 6])
        else:
            shared_start_p0_eci[shared_start_id] = n.asarray(
                fixed_start_positions[shared_start_id],
                dtype=float,
            ).copy()
            shared_start_v0_eci[shared_start_id] = n.asarray(
                final_stage["xopt"][group_base:group_base + 3],
                dtype=float,
            ).copy()
            shared_start_log_B0[shared_start_id] = float(final_stage["xopt"][group_base + 3])
    shared_start_B0 = {
        shared_start_id: float(10.0 ** shared_log_B0)
        for shared_start_id, shared_log_B0 in shared_start_log_B0.items()
    }

    if verbose > 0:
        print("fitted shared start states and ballistic coefficients:")
        for shared_start_id in final_stage["shared_start_order"]:
            print(
                "  start fragment %s: p0=%s v0=%s log10(B0)=%1.6f B0=%1.6e m^2/kg"
                % (
                    shared_start_id,
                    n.array2string(shared_start_p0_eci[shared_start_id], precision=3),
                    n.array2string(shared_start_v0_eci[shared_start_id], precision=3),
                    shared_start_log_B0[shared_start_id],
                    shared_start_B0[shared_start_id],
                )
            )

    return {
        "path_results": path_results,
        "shared_start_order": tuple(final_stage["shared_start_order"]),
        "shared_start_p0_eci": shared_start_p0_eci,
        "shared_start_v0_eci": shared_start_v0_eci,
        "shared_start_log_B0": shared_start_log_B0,
        "shared_start_B0": shared_start_B0,
        "optimizer": dict(final_stage["optimizer"]),
        "optimizer_zero_wind": dict(zero_wind_stage["optimizer"]),
        "optimizer_era5": None if era5_stage is None else dict(era5_stage["optimizer"]),
        "wind_model_info": list(final_wind_model_info),
    }


def impact_observables_enu(impact, ref_ecef, ref_lat_deg, ref_lon_deg):
    impact_pos_ecef = eci_to_ecef_position(
        impact["impact_pos_eci"],
        impact["impact_time_unix"],
    )
    lat_rad = n.deg2rad(ref_lat_deg)
    lon_rad = n.deg2rad(ref_lon_deg)

    rot = n.array(
        [
            [-n.sin(lon_rad), n.cos(lon_rad), 0.0],
            [-n.sin(lat_rad) * n.cos(lon_rad), -n.sin(lat_rad) * n.sin(lon_rad), n.cos(lat_rad)],
            [n.cos(lat_rad) * n.cos(lon_rad), n.cos(lat_rad) * n.sin(lon_rad), n.sin(lat_rad)],
        ],
        dtype=float,
    )
    delta_enu = rot @ (impact_pos_ecef - ref_ecef)
    return n.array(
        [
            float(impact["impact_time_unix"]),
            float(delta_enu[0]),
            float(delta_enu[1]),
        ],
        dtype=float,
    )


def estimate_impact_uncertainty(xhat, covariance_info, t, dt_model, density_profile=None, wind_model=None):
    if covariance_info is None:
        return None

    nominal_model = build_model_from_parameters(
        xhat,
        t,
        dt_model,
        density_profile=density_profile,
        wind_model=wind_model,
    )
    nominal_result = {
        "model": nominal_model,
        "times_unix": t,
        "dt_model": dt_model,
    }
    nominal_impact = extrapolate_best_fit_to_ground(nominal_result)
    nominal_ecef = eci_to_ecef_position(
        nominal_impact["impact_pos_eci"],
        nominal_impact["impact_time_unix"],
    )

    def impact_map(x):
        model = build_model_from_parameters(
            x,
            t,
            dt_model,
            density_profile=density_profile,
            wind_model=wind_model,
        )
        impact = extrapolate_best_fit_to_ground(
            {
                "model": model,
                "times_unix": t,
                "dt_model": dt_model,
            }
        )
        return impact_observables_enu(
            impact,
            nominal_ecef,
            nominal_impact["impact_lat_deg"],
            nominal_impact["impact_lon_deg"],
        )

    impact_jac = finite_difference_jacobian(
        impact_map,
        xhat,
        parameter_step_sizes(xhat),
    )
    impact_cov = impact_jac @ covariance_info["covariance"] @ impact_jac.T

    time_std = float(n.sqrt(max(impact_cov[0, 0], 0.0)))
    horizontal_cov = impact_cov[1:, 1:]
    horizontal_std = n.sqrt(n.maximum(n.diag(horizontal_cov), 0.0))

    eigvals, eigvecs = n.linalg.eigh(horizontal_cov)
    eigvals = n.maximum(eigvals, 0.0)
    order = n.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_axis = float(n.sqrt(eigvals[-1]))
    minor_axis = float(n.sqrt(eigvals[0]))
    major_vec = eigvecs[:, -1]
    major_azimuth = float((n.degrees(n.arctan2(major_vec[0], major_vec[1])) + 360.0) % 360.0)

    return {
        "impact_time_std_s": time_std,
        "impact_east_std_m": float(horizontal_std[0]),
        "impact_north_std_m": float(horizontal_std[1]),
        "impact_horizontal_cov_enu_m2": horizontal_cov,
        "impact_horizontal_major_axis_1sigma_m": major_axis,
        "impact_horizontal_minor_axis_1sigma_m": minor_axis,
        "impact_horizontal_major_axis_azimuth_deg": major_azimuth,
        "impact_observable_jacobian": impact_jac,
        "parameter_covariance": covariance_info["covariance"],
        "parameter_std": covariance_info["std"],
        "residual_rss": covariance_info["rss"],
        "residual_dof": covariance_info["dof"],
        "residual_variance": covariance_info["sigma2"],
    }


def fit_shared_ballistic_coefficient(
                                    fragment_pos,
                                    fragment_pos_err,
                                    fragment_times,
                                    fit_ids,
                                    gamma=0.5,
                                    B0_guess=[-3, -3, -3],
                                    plot_context_fragment_geo_pos=None,
                                    debug_plot=False,
                                    debug_plot_every=100,
                                    verbose=2):
    # Gamma is held fixed during the fit. The optimized parameters are the
    # initial ECI state and a set of log10 ballistic-coefficient node values.
    gamma_fixed = float(gamma)

    t = n.asarray(fragment_times, dtype=float).reshape(-1)
    pos_ecef = n.asarray(fragment_pos, dtype=float)
    pos_ecef_err = n.asarray(fragment_pos_err, dtype=float).reshape(-1)

    if pos_ecef.ndim != 2 or pos_ecef.shape[1] != 3:
        raise ValueError("fragment_pos must have shape (n, 3)")
    if t.shape[0] != pos_ecef.shape[0]:
        raise ValueError("fragment_times and fragment_pos must have the same length")
    if pos_ecef_err.shape[0] != pos_ecef.shape[0]:
        raise ValueError("fragment_pos_err and fragment_pos must have the same length")
    if t.size < 3:
        raise ValueError("Need at least three 3D measurements for the fit.")

    order = n.argsort(t, kind="mergesort")
    t = t[order]
    pos_ecef = pos_ecef[order, :]
    pos_ecef_err = pos_ecef_err[order]

    p0_guess_ecef = pos_ecef[0, :]
    dt_obs = t[-1] - t[0]
    if dt_obs <= 0.0:
        raise ValueError("Need observations spanning a non-zero time interval.")
    v0_guess_ecef = estimate_initial_velocity_ecef(t, pos_ecef)

    pos_eci = ecef_to_eci_position(pos_ecef, t)
    density_profile = build_msis_density_profile(t, pos_ecef)
    p0_guess_eci, v0_guess_eci = ecef_to_eci_state(
        p0_guess_ecef,
        v0_guess_ecef,
        t[0],
    )

    if verbose > 1:
        print("v0_guess_ecef", v0_guess_ecef)
        print("p0_guess_ecef", p0_guess_ecef)
        print("v0_guess_eci", v0_guess_eci)
        print("p0_guess_eci", p0_guess_eci)

    B0_guess = n.asarray(B0_guess, dtype=float).reshape(-1)
    if B0_guess.size < 2:
        raise ValueError("B0_guess must contain at least two node values.")

    x0 = n.concatenate((p0_guess_eci, v0_guess_eci, B0_guess))
    if pos_eci.size < x0.size:
        raise ValueError("Need at least three 3D measurements for the fit.")

    dt_model = 2.0#min(0.5, max((n.max(t) - n.min(t)) / 200.0, 0.05))
    zero_wind_stage = run_fmin_fit_stage(
        x0,
        t,
        pos_eci,
        dt_model,
        gamma_fixed,
        density_profile=density_profile,
        wind_model=None,
        stage_name="zero_wind",
        debug_plot=debug_plot,
        debug_plot_every=debug_plot_every,
        verbose=verbose,
    )
    zero_wind_result = {
        "model": zero_wind_stage["model"],
        "times_unix": t,
        "dt_model": dt_model,
    }

    wind_model = None
    wind_model_info = {
        "type": "corotating_fallback",
        "reason": "ERA5 refinement not attempted",
    }
    era5_stage = None
    try:
        wind_model, wind_model_info = build_cached_era5_wind_model_from_result(
            zero_wind_result,
            max_time_ahead=3600.0,
            verbose=verbose,
        )
        if wind_model is not None:
            if verbose > 0:
                print("rerunning fit with cached ERA5 wind profile")
            era5_stage = run_fmin_fit_stage(
                zero_wind_stage["xopt"],
                t,
                pos_eci,
                dt_model,
                gamma_fixed,
                density_profile=density_profile,
                wind_model=wind_model,
                stage_name="era5_profile",
                debug_plot=debug_plot,
                debug_plot_every=debug_plot_every,
                verbose=verbose,
            )
    except Exception as exc:
        wind_model = None
        wind_model_info = {
            "type": "corotating_fallback",
            "error": str(exc),
        }

    final_stage = zero_wind_stage if era5_stage is None else era5_stage
    optimizer = final_stage["optimizer"]
    covariance_info = final_stage["covariance_info"]
    xhat = n.asarray(final_stage["xopt"], dtype=float).copy()

    p0_hat_eci = xhat[0:3]
    v0_hat_eci = xhat[3:6]
    B0 = xhat[6:]
    model = final_stage["model"]

    result = {
        "times_unix": t,
        "pos_ecef": pos_ecef,
        "pos_ecef_err": pos_ecef_err,
        "pos_eci": pos_eci,
        "fit_ids": tuple(str(fid) for fid in fit_ids),
        "p0_guess_ecef": p0_guess_ecef,
        "v0_guess_ecef": v0_guess_ecef,
        "p0_guess_eci": p0_guess_eci,
        "v0_guess_eci": v0_guess_eci,
        "p0_hat_eci": p0_hat_eci,
        "v0_hat_eci": v0_hat_eci,
        "B0_hat": B0,
        "B_node_times_unix": model.get("B_node_times_unix"),
        "fit_stage": optimizer["stage"],
        "dt_model": dt_model,
        "optimizer": optimizer,
        "optimizer_zero_wind": zero_wind_stage["optimizer"],
        "optimizer_era5": None if era5_stage is None else era5_stage["optimizer"],
        "parameter_covariance": None if covariance_info is None else covariance_info["covariance"],
        "parameter_std": None if covariance_info is None else covariance_info["std"],
        "density_profile_altitude_m": density_profile["altitude_grid_m"],
        "density_profile_rho_kg_m3": density_profile["rho_grid_kg_m3"],
        "density_profile_reference_time_unix": density_profile["reference_time_unix"],
        "density_profile_reference_lat_deg": density_profile["reference_lat_deg"],
        "density_profile_reference_lon_deg": density_profile["reference_lon_deg"],
        "wind_model_info": wind_model_info,
        "specific_energy_loss_rate_w_kg": model["specific_energy_loss_rate_w_kg"],
        "specific_energy_loss_rate_interp": model["specific_energy_loss_rate_interp"],
        "speed_m_s": model["speed_m_s"],
        "speed_interp": model["speed_interp"],
        "relative_speed_m_s": model["relative_speed_m_s"],
        "relative_speed_interp": model["relative_speed_interp"],
        "fit_parameter_names": tuple(
            [
                "p0_eci_x", "p0_eci_y", "p0_eci_z",
                "v0_eci_x", "v0_eci_y", "v0_eci_z",
            ]
            + [f"log_B{i:02d}" for i in range(B0.size)]
        ),
        "model": model,
    }

    try:
        result["impact"] = extrapolate_best_fit_to_ground(result)
    except ValueError as exc:
        result["impact"] = None
        result["impact_error"] = str(exc)
    else:
        try:
            result["impact_uncertainty"] = estimate_impact_uncertainty(
                xhat,
                covariance_info,
                t,
                dt_model,
                density_profile=density_profile,
                wind_model=wind_model,
            )
        except ValueError as exc:
            result["impact_uncertainty"] = None
            result["impact_uncertainty_error"] = str(exc)
        except Exception as exc:
            result["impact_uncertainty"] = None
            result["impact_uncertainty_error"] = str(exc)

    if "impact_uncertainty" not in result:
        result["impact_uncertainty"] = None

    hdf5_path = save_result_to_hdf5(result, fit_ids)
    result["hdf5_path"] = str(hdf5_path)
    if plot_context_fragment_geo_pos is not None:
        result["plot_context_fragment_geo_pos"] = plot_context_fragment_geo_pos

    if verbose > 0:
        print("p0_hat_eci", result["p0_hat_eci"])
        print("v0_hat_eci", result["v0_hat_eci"])
        #print("B0_hat", result["B0_hat"])
        #print("fr_hat", result["fr_hat"])
        print("fit_stage", result["fit_stage"])
        print("zero_wind_cost", zero_wind_stage["optimizer"]["fun"])
        if era5_stage is not None:
            print("era5_cost", era5_stage["optimizer"]["fun"])
        print("cost", optimizer["fun"])
        print("success", optimizer["success"])
        print("hdf5_path", result["hdf5_path"])
        if result["impact"] is not None:
            print("impact_lat_deg", result["impact"]["impact_lat_deg"])
            print("impact_lon_deg", result["impact"]["impact_lon_deg"])
            print("impact_time_unix", result["impact"]["impact_time_unix"])
            if result["impact_uncertainty"] is not None:
                print("impact_time_std_s", result["impact_uncertainty"]["impact_time_std_s"])
                print("impact_east_std_m", result["impact_uncertainty"]["impact_east_std_m"])
                print("impact_north_std_m", result["impact_uncertainty"]["impact_north_std_m"])
                print(
                    "impact_horizontal_major_axis_1sigma_m",
                    result["impact_uncertainty"]["impact_horizontal_major_axis_1sigma_m"],
                )
                print(
                    "impact_horizontal_minor_axis_1sigma_m",
                    result["impact_uncertainty"]["impact_horizontal_minor_axis_1sigma_m"],
                )
            elif "impact_uncertainty_error" in result:
                print("impact_uncertainty_error", result["impact_uncertainty_error"])
        elif "impact_error" in result:
            print("impact_error", result["impact_error"])
        plot_ballistic_fit(result, show=bool(verbose > 1))
        #plot_density_profile(result)
        #plot_specific_energy_loss_rate(result)
        #plot_velocity_scatter(result)

    return result

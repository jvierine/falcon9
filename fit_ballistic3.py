import numpy as n
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.interpolate as sint
import scipy.optimize as so
import importlib.util
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


def build_fit_result_hdf5_path(fit_ids):
    fit_ids = list(fit_ids)
    if len(fit_ids) == 0:
        ids_part = "none"
    else:
        ids_part = "_".join(_sanitize_filename_token(fid) for fid in fit_ids)
    return Path(__file__).with_name(f"ballistic_fit_{ids_part}.h5")


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


def save_result_to_hdf5(result, fit_ids):
    import h5py

    out_path = build_fit_result_hdf5_path(fit_ids)
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
            geomagnetic_activity=-1
        )

        arr = n.asarray(data)

        # pymsis total mass density is typically the first species/output entry.
        # Squeeze to make indexing robust across wrapper return shapes.
        arr = n.squeeze(arr)
        rho_a[j] = arr[0] if n.ndim(arr) > 0 else arr

    return rho_a

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


def sigmoid(x):
    return 1.0 / (1.0 + n.exp(-x))




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


def state_to_geodetic_density(pos_eci, unix_time):
    lat, lon, hgt = eci_to_geodetic(pos_eci, unix_time)
    hgt_msis = float(n.clip(hgt, 0.0, 300e3))
    rho_a = get_msis_density(
        n.array([unix_time], dtype=float),
        n.array([lat], dtype=float),
        n.array([lon], dtype=float),
        n.array([hgt_msis], dtype=float),
    )[0]
    return float(lat), float(lon), float(hgt), float(rho_a)


def drag_state_eci(pos_eci, vel_eci, unix_time, B_now):
    """
    Evaluate local atmospheric state, drag acceleration, and specific drag
    power at a single ECI state.
    """
    lat, lon, hgt, rho_a = state_to_geodetic_density(pos_eci, unix_time)
    v_rel = vel_eci - atmosphere_velocity_eci(pos_eci)
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


def plot_ballistic_fit(result):
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(lon_model, lat_model, "-", lw=2, label="Best fit model")
    axes[0, 0].plot(lon_meas, lat_meas, ".", ms=5, label="Measurements")
    for i, (frag_id, lat_frag, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        axes[0, 0].plot(
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
        axes[0, 0].text(
            lon_frag + 0.02,
            lat_frag + 0.02,
            frag_id,
            fontsize=7,
            ha="left",
            va="bottom",
            zorder=17,
        )
    axes[0, 0].set_xlabel("Longitude (deg)")
    axes[0, 0].set_ylabel("Latitude (deg)")
    axes[0, 0].legend()

    axes[0, 1].plot(t_model - t0, hgt_model / 1e3, "-", lw=2, label="Best fit model")
    axes[0, 1].plot(t_meas - t0, hgt_meas / 1e3, ".", ms=5, label="Measurements")
    axes[0, 1].set_xlabel("Time since first sample (s)")
    axes[0, 1].set_ylabel("Height (km)")
    axes[0, 1].legend()

    axes[1, 0].semilogy(t_model - t0, B_model, "-", lw=2, label="Best fit model")
    axes[1, 0].set_xlabel("Time since first sample (s)")
    axes[1, 0].set_ylabel("B")
    axes[1, 0].legend()

    axes[1, 1].plot(lon_model, hgt_model / 1e3, "-", lw=2, label="Best fit model")
    axes[1, 1].plot(lon_meas, hgt_meas / 1e3, ".", ms=5, label="Measurements")
    for i, (_, _, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        axes[1, 1].axvline(
            lon_frag,
            color="red",
            linestyle="--",
            linewidth=0.8,
            zorder=8,
            label="Recovered fragment longitude" if i == 0 else None,
        )
    axes[1, 1].set_xlabel("Longitude (deg)")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].legend()

    if impact is not None:
        impact_model = impact["trajectory"]
        impact_order = n.argsort(impact_model["times_model"])
        t_impact = impact_model["times_model"][impact_order]
        lat_impact = impact_model["lat_deg"][impact_order]
        lon_impact = impact_model["lon_deg"][impact_order]
        hgt_impact = impact_model["hgt_m"][impact_order]
        B_impact = impact_model["B_model"][impact_order]

        axes[0, 0].plot(lon_impact, lat_impact, "--", lw=2, label="Extrapolated path")
        axes[0, 0].plot(
            impact["impact_lon_deg"],
            impact["impact_lat_deg"],
            "x",
            ms=8,
            mew=2,
            label="Impact",
        )
        axes[0, 0].legend()

        axes[0, 1].plot(t_impact - t0, hgt_impact / 1e3, "--", lw=2, label="Extrapolated path")
        axes[0, 1].plot(
            impact["impact_time_unix"] - t0,
            impact["impact_hgt_m"] / 1e3,
            "x",
            ms=8,
            mew=2,
            label="Impact",
        )
        axes[0, 1].legend()

        axes[1, 0].semilogy(t_impact - t0, B_impact, "--", lw=2, label="Extrapolated path")
        axes[1, 0].legend()

        axes[1, 1].plot(lon_impact, hgt_impact / 1e3, "--", lw=2, label="Extrapolated path")
        axes[1, 1].plot(
            impact["impact_lon_deg"],
            impact["impact_hgt_m"] / 1e3,
            "x",
            ms=8,
            mew=2,
            label="Impact",
        )
        axes[1, 1].legend()

    fig.tight_layout()
    plt.show()


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



def collect_data(fid,fpos,ftimes,ids):
    pos=[]
    tv=[]
    for i in range(len(fid)):
        if fid[i] in ids:
            pos.append(fpos[i])        
            tv.append(ftimes[i])
    tv=n.concatenate(tv)
    pos=n.concatenate(pos)
    order = n.argsort(tv, kind="mergesort")
    tv = n.asarray(tv[order], dtype=float)
    pos = n.asarray(pos[order], dtype=float)
    print(pos.shape)
    print(tv.shape)
    return(tv,pos)


def collect_data_merged(fid, fpos, ftimes, ids):
    """
    Collect and sort the selected fragment measurements, then merge samples
    that fall within the same integer Unix second by averaging their times and
    ECEF positions.
    """
    tv, pos = collect_data(fid, fpos, ftimes, ids)
    if tv.size == 0:
        return tv, pos

    second_bins = n.floor(tv).astype(n.int64)
    unique_seconds, first_idx, counts = n.unique(
        second_bins,
        return_index=True,
        return_counts=True,
    )

    t_merged = n.empty(unique_seconds.size, dtype=float)
    pos_merged = n.empty((unique_seconds.size, pos.shape[1]), dtype=float)

    for i, (start, count) in enumerate(zip(first_idx, counts)):
        sl = slice(start, start + count)
        t_merged[i] = n.mean(tv[sl])
        pos_merged[i] = n.mean(pos[sl], axis=0)

    print(pos_merged.shape)
    print(t_merged.shape)
    return t_merged, pos_merged


def propagate(
    p0,
    v0,
    t,
    B0=[-3,-3,-3],
    dt=0.5,
    fixed_B=None,
    start_time=None,
    stop_at_ground=False,
):
    """
    Simple forward propagation utility used for quick inspection/debugging.
    """
    t = n.asarray(t, dtype=float)
    if t.size == 0:
        raise ValueError("Need at least one observation time to propagate.")

    if fixed_B is None:
        mtv = n.linspace(n.min(t) - 2 * dt, n.max(t) + 2 * dt, len(B0))
        Bfun = sint.interp1d(mtv, B0)
    else:
        B_const = fixed_B

        def Bfun(t_query):
            t_arr = n.asarray(t_query, dtype=float)
            if t_arr.ndim == 0:
                return B_const
            return n.full(t_arr.shape, B_const, dtype=float)
    print(B0)
    p = n.asarray(p0, dtype=float).copy()
    v = n.asarray(v0, dtype=float).copy()
    if start_time is None:
        tnow = n.min(t) - dt
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


def residuals_lm(x, t, pos_eci, dt_model, gamma_fixed):
    p0 = x[0:3]
    v0 = x[3:6]
    B0 = x[6:8]
#    C_L=x[8]
#    fr_raw = n.clip(x[7:9], -10.0, 10.0)

    # propagate() currently uses the 0.5 drag prefactor internally, so scale
    # B0 if the caller wants to keep a different fixed gamma convention.
    #B0_eff = B0 * (gamma_fixed / 0.5)
    model = propagate(p0, v0, t, B0=B0, dt=dt_model)
    model_pos = model["pos_eci_interp"](t)
    residuals = model_pos - pos_eci

    # Give the first 100 seconds 100x weight in the least-squares objective.
    rel_t = t - n.min(t)
    point_weights = n.ones_like(rel_t, dtype=float)
    #point_weights[rel_t <= 30.0] = n.sqrt(1000.0)
#    point_weights[rel_t <= 20.0] = n.sqrt(1000.0)
 #   point_weights[rel_t <= 20.0] = n.sqrt(1000.0)

    return (residuals * point_weights[:, None]).ravel()


def objective_fmin(x, t, pos_eci, dt_model, gamma_fixed, fr_penalty_weight=1e6):
    residuals = residuals_lm(x, t, pos_eci, dt_model, gamma_fixed)
    #overflow = n.maximum(n.abs(x[7:9]) - 100.0, 0.0)
    #penalty = fr_penalty_weight * n.sum(overflow**2)
    s= float(n.sum(residuals**2))
    print(s)
    return(s)


def fit_shared_ballistic_coefficient(
                                    fragment_ids,
                                    fragment_pos,
                                    fragment_times,
                                    fit_ids,
                                    gamma=0.5,
                                    B0_guess=[-3,-3],
                                    verbose=2):
    # Gamma is held fixed during the fit. The optimized parameters are the
    # initial ECI state, B0, fr1, and fr2.
    gamma_fixed = float(gamma)

    t, pos_ecef = collect_data_merged(fragment_ids, fragment_pos, fragment_times, fit_ids)

    p0_guess_ecef = pos_ecef[0, :]
    dt_obs = t[-1] - t[0]
    v0_guess_ecef = (pos_ecef[-1, :] - pos_ecef[0, :]) / dt_obs

    pos_eci = ecef_to_eci_position(pos_ecef, t)
    p0_guess_eci, v0_guess_eci = ecef_to_eci_state(
        p0_guess_ecef,
        v0_guess_ecef,
        t[0],
    )

    print("v0_guess_ecef", v0_guess_ecef)
    print("p0_guess_ecef", p0_guess_ecef)
    print("v0_guess_eci", v0_guess_eci)
    print("p0_guess_eci", p0_guess_eci)
#    exit(0)

    x0 = n.concatenate((
        p0_guess_eci,
        v0_guess_eci,
        [-3,-3],
        [-1],
    ))
    if pos_eci.size < x0.size:
        raise ValueError("Need at least three 3D measurements for the fit.")

    dt_model = 0.5#min(0.5, max((n.max(t) - n.min(t)) / 200.0, 0.05))
    xopt, fopt, iterations, funcalls, warnflag = so.fmin(
        objective_fmin,
        x0,
        args=(t, pos_eci, dt_model, gamma_fixed),
        full_output=True,
        maxiter=4000,
        maxfun=10000,
        disp=bool(verbose > 1),
    )
    optimizer = {
        "method": "fmin",
        "x": xopt,
        "fun": float(fopt),
        "nit": int(iterations),
        "nfev": int(funcalls),
        "warnflag": int(warnflag),
        "success": bool(warnflag == 0),
    }

    xhat = n.asarray(xopt, dtype=float).copy()
    #xhat[7:9] = n.clip(xhat[7:9], -10.0, 10.0)

    p0_hat_eci = xhat[0:3]
    v0_hat_eci = xhat[3:6]
    B0 = xhat[6:8]
#    fr_raw_hat = xhat[7:9]
    #fr_hat = sigmoid(fr_raw_hat)

    #B0_hat_eff = B0_hat * (gamma_fixed / 0.5)
    model = propagate(p0_hat_eci, v0_hat_eci, t, B0, dt=dt_model)

    result = {
        "times_unix": t,
        "pos_ecef": pos_ecef,
        "pos_eci": pos_eci,
        "p0_guess_ecef": p0_guess_ecef,
        "v0_guess_ecef": v0_guess_ecef,
        "p0_guess_eci": p0_guess_eci,
        "v0_guess_eci": v0_guess_eci,
        "p0_hat_eci": p0_hat_eci,
        "v0_hat_eci": v0_hat_eci,
        "B0_hat": B0,
#        "B0_hat_effective": B0_hat_eff,
 #       "fr_raw_hat": fr_raw_hat,
  #      "fr_hat": fr_hat,
   #     "gamma_fixed": gamma_fixed,
        "dt_model": dt_model,
        "optimizer": optimizer,
        "specific_energy_loss_rate_w_kg": model["specific_energy_loss_rate_w_kg"],
        "specific_energy_loss_rate_interp": model["specific_energy_loss_rate_interp"],
        "speed_m_s": model["speed_m_s"],
        "speed_interp": model["speed_interp"],
        "relative_speed_m_s": model["relative_speed_m_s"],
        "relative_speed_interp": model["relative_speed_interp"],
        "fit_parameter_names": (
            "p0_eci_x", "p0_eci_y", "p0_eci_z",
            "v0_eci_x", "v0_eci_y", "v0_eci_z",
            "log_B00", "log_B01", "log_B02", "C_L",
        ),
        "model": model,
    }

    try:
        result["impact"] = extrapolate_best_fit_to_ground(result)
    except ValueError as exc:
        result["impact"] = None
#        result["impact_error"] = str(exc)

    hdf5_path = save_result_to_hdf5(result, fit_ids)
    result["hdf5_path"] = str(hdf5_path)

    if verbose > 0:
        print("p0_hat_eci", result["p0_hat_eci"])
        print("v0_hat_eci", result["v0_hat_eci"])
        #print("B0_hat", result["B0_hat"])
        #print("fr_hat", result["fr_hat"])
        print("cost", optimizer["fun"])
        print("success", optimizer["success"])
        print("hdf5_path", result["hdf5_path"])
        if result["impact"] is not None:
            print("impact_lat_deg", result["impact"]["impact_lat_deg"])
            print("impact_lon_deg", result["impact"]["impact_lon_deg"])
            print("impact_time_unix", result["impact"]["impact_time_unix"])
        elif "impact_error" in result:
            print("impact_error", result["impact_error"])
        plot_ballistic_fit(result)
        plot_density_profile(result)
        plot_specific_energy_loss_rate(result)
        plot_velocity_scatter(result)

    return result

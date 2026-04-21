from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


G0 = 9.80665
R_D = 287.06
OMEGA_EARTH = 7.2921150e-5  # rad/s
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_MEAN_RADIUS_M = 6371000.0
DEFAULT_MODEL_LEVELS = "1/to/137"
DEFAULT_PRESSURE_LEVELS_HPA = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


def unix_to_datetime64(unix_time):
    dt = datetime.fromtimestamp(float(unix_time), tz=timezone.utc)
    return np.datetime64(dt.replace(tzinfo=None), "ns")


def geopotential_to_geometric_height(geopotential_m2_s2):
    geopotential_height = np.asarray(geopotential_m2_s2, dtype=float) / G0
    return (
        EARTH_MEAN_RADIUS_M * geopotential_height
        / np.maximum(EARTH_MEAN_RADIUS_M - geopotential_height, 1.0)
    )


def gmst_angle(unix_time):
    unix_time = np.asarray(unix_time, dtype=float)
    jd = unix_time / 86400.0 + 2440587.5
    t_ut1 = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t_ut1**2
        - t_ut1**3 / 38710000.0
    )
    return np.deg2rad(np.mod(gmst_deg, 360.0))


def geodetic_to_ecef(lat_deg, lon_deg, hgt_m):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    hgt_m = np.asarray(hgt_m, dtype=float)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (N + hgt_m) * cos_lat * cos_lon
    y = (N + hgt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + hgt_m) * sin_lat
    return np.stack((x, y, z), axis=-1)


def enu_to_ecef_vector(east_mps, north_mps, up_mps, lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    east_hat = np.stack((-sin_lon, cos_lon, np.zeros_like(sin_lon)), axis=-1)
    north_hat = np.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
        axis=-1,
    )
    up_hat = np.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), axis=-1)

    east_mps = np.asarray(east_mps, dtype=float)[..., None]
    north_mps = np.asarray(north_mps, dtype=float)[..., None]
    up_mps = np.asarray(up_mps, dtype=float)[..., None]
    return east_mps * east_hat + north_mps * north_hat + up_mps * up_hat


def ecef_to_eci_vector(vec_ecef, unix_time):
    vec_ecef = np.asarray(vec_ecef, dtype=float)
    theta = gmst_angle(unix_time)
    c = np.cos(theta)
    s = np.sin(theta)

    x = vec_ecef[..., 0]
    y = vec_ecef[..., 1]
    z = vec_ecef[..., 2]
    return np.stack((c * x - s * y, s * x + c * y, z), axis=-1)


def ensure_era5_pressure_level_file(
    target_path,
    start_time_unix,
    end_time_unix=None,
    area=None,
    pressure_levels_hpa=None,
    overwrite=False,
):
    """
    Download an ERA5 pressure-level NetCDF containing u, v, and geopotential.

    Parameters
    ----------
    target_path : str or Path
        Output NetCDF path.
    start_time_unix, end_time_unix : float
        UTC unix timestamps. ERA5 is hourly, so the request rounds out to whole
        hours spanning the supplied interval.
    area : sequence[float], optional
        [north, west, south, east] in degrees.
    pressure_levels_hpa : sequence[int], optional
        Pressure levels to request. Defaults to the full ERA5 pressure-level set.
    """
    target_path = Path(target_path)
    if target_path.exists() and not overwrite:
        return target_path

    if end_time_unix is None:
        end_time_unix = start_time_unix

    pressure_levels_hpa = pressure_levels_hpa or DEFAULT_PRESSURE_LEVELS_HPA

    start_dt = datetime.fromtimestamp(float(start_time_unix), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(float(end_time_unix), tz=timezone.utc)
    start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    end_dt = end_dt.replace(minute=0, second=0, microsecond=0)
    if end_dt < start_dt:
        raise ValueError("end_time_unix must be greater than or equal to start_time_unix")

    request_hours = []
    request_days = set()
    cursor = start_dt
    while cursor <= end_dt:
        request_hours.append(cursor.strftime("%H:00"))
        request_days.add(cursor.date())
        cursor += timedelta(hours=1)

    years = sorted({day.strftime("%Y") for day in request_days})
    months = sorted({day.strftime("%m") for day in request_days})
    days = sorted({day.strftime("%d") for day in request_days})

    request = {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": [str(level) for level in pressure_levels_hpa],
        "year": years,
        "month": months,
        "day": days,
        "time": sorted(set(request_hours)),
        "format": "netcdf",
    }
    if area is not None:
        request["area"] = [float(v) for v in area]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve("reanalysis-era5-pressure-levels", request, str(target_path))
    return target_path


def _build_complete_request_times(start_time_unix, end_time_unix=None):
    if end_time_unix is None:
        end_time_unix = start_time_unix

    start_dt = datetime.fromtimestamp(float(start_time_unix), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(float(end_time_unix), tz=timezone.utc)
    start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    end_dt = end_dt.replace(minute=0, second=0, microsecond=0)
    if end_dt < start_dt:
        raise ValueError("end_time_unix must be greater than or equal to start_time_unix")

    request_hours = []
    request_days = set()
    cursor = start_dt
    while cursor <= end_dt:
        request_hours.append(cursor.strftime("%H:%M:%S"))
        request_days.add(cursor.date())
        cursor += timedelta(hours=1)

    return {
        "dates": sorted(day.strftime("%Y-%m-%d") for day in request_days),
        "times": sorted(set(request_hours)),
    }


def ensure_era5_model_level_files(
    target_prefix,
    start_time_unix,
    end_time_unix=None,
    area=None,
    model_levels=DEFAULT_MODEL_LEVELS,
    overwrite=False,
):
    """
    Download ERA5 complete model-level GRIB files needed for high-altitude wind.

    This retrieves:
    - t, q, u, v on model levels 1..137
    - z and lnsp on model level 1
    """
    target_prefix = Path(target_prefix)
    tq_uv_path = target_prefix.with_name(f"{target_prefix.name}_tquv.grib")
    z_lnsp_path = target_prefix.with_name(f"{target_prefix.name}_zlnsp.grib")

    if tq_uv_path.exists() and z_lnsp_path.exists() and not overwrite:
        return tq_uv_path, z_lnsp_path

    time_spec = _build_complete_request_times(start_time_unix, end_time_unix=end_time_unix)
    request = {
        "class": "ea",
        "expver": "1",
        "levtype": "ml",
        "stream": "oper",
        "type": "an",
        "date": "/".join(time_spec["dates"]),
        "time": "/".join(time_spec["times"]),
    }
    if area is not None:
        request["area"] = [float(v) for v in area]
        request["grid"] = "0.25/0.25"

    tq_uv_request = dict(request)
    tq_uv_request.update(
        {
            "levelist": str(model_levels),
            "param": "130/131/132/133",
        }
    )

    z_lnsp_request = dict(request)
    z_lnsp_request.update(
        {
            "levelist": "1",
            "param": "129/152",
        }
    )

    tq_uv_path.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-complete", tq_uv_request, str(tq_uv_path))
    client.retrieve("reanalysis-era5-complete", z_lnsp_request, str(z_lnsp_path))
    return tq_uv_path, z_lnsp_path


def _sort_dataset_coords(dataset, coord_names):
    ds = dataset
    for coord_name in coord_names:
        if coord_name in ds.coords and ds[coord_name].ndim > 0:
            ds = ds.sortby(coord_name)
    return ds


def _normalise_longitude(dataset, lon_deg):
    lon = float(lon_deg)
    lon_vals = np.asarray(dataset["longitude"].values, dtype=float)
    if lon_vals.min() >= 0.0 and lon_vals.max() > 180.0:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


def _time_coord_name(dataset):
    for name in ("time", "valid_time"):
        if name in dataset.coords and dataset[name].ndim > 0:
            return name
    return None


def _interp_dataset_point(dataset, lat_deg, lon_deg, unix_time=None):
    coords = {
        "latitude": float(lat_deg),
        "longitude": _normalise_longitude(dataset, lon_deg),
    }
    time_coord = _time_coord_name(dataset)
    if time_coord is not None:
        if unix_time is None:
            raise ValueError("unix_time is required when interpolating a time-varying ERA5 dataset")
        time_query = np.asarray([unix_to_datetime64(unix_time)]).astype(dataset[time_coord].dtype)[0]
        coords[time_coord] = time_query
    return dataset.interp(coords, method="linear")


def _get_model_level_pv(grib_path):
    from eccodes import codes_get_array, codes_grib_new_from_file, codes_release

    with open(grib_path, "rb") as handle:
        gid = codes_grib_new_from_file(handle)
        if gid is None:
            raise ValueError(f"Could not read GRIB message from {grib_path}")
        pv = np.asarray(codes_get_array(gid, "pv"), dtype=float)
        codes_release(gid)
    return pv


def _compute_model_level_geopotential_profiles(t_profile, q_profile, surface_geopotential, lnsp, pv):
    """
    Compute geopotential on ERA5 model full levels for one atmospheric column.

    The algorithm follows ECMWF's official ERA5 model-level hydrostatic recipe:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=158636068
    """
    t_profile = np.asarray(t_profile, dtype=float)
    q_profile = np.asarray(q_profile, dtype=float)
    surface_geopotential = np.asarray(surface_geopotential, dtype=float)
    lnsp = np.asarray(lnsp, dtype=float)

    sp = np.exp(lnsp)
    nlevels = t_profile.shape[0]
    a = pv[: nlevels + 1]
    b = pv[nlevels + 1 :]
    if len(a) != nlevels + 1 or len(b) != nlevels + 1:
        raise ValueError("Unexpected ERA5 model-level pv coefficient length")

    z_full = np.empty_like(t_profile, dtype=float)
    p_half = np.empty(nlevels + 1, dtype=float)
    p_half[:] = a + b * sp

    z_h = float(surface_geopotential)
    for idx in range(nlevels - 1, -1, -1):
        t_level = t_profile[idx] * (1.0 + 0.609133 * q_profile[idx])
        ph_lev = p_half[idx]
        ph_levplusone = p_half[idx + 1]

        if idx == 0:
            dlog_p = np.log(ph_levplusone / 0.1)
            alpha = np.log(2.0)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1.0 - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        t_level = t_level * R_D
        z_full[idx] = z_h + (t_level * alpha)
        z_h = z_h + (t_level * dlog_p)

    return z_full


@dataclass
class ERA5Wind:
    dataset: xr.Dataset

    def __post_init__(self):
        self.dataset = _sort_dataset_coords(
            self.dataset,
            ("time", "valid_time", "latitude", "longitude", "pressure_level", "level"),
        )

    @classmethod
    def open(cls, path):
        dataset = xr.open_dataset(path).load()
        return cls(dataset=dataset)

    @classmethod
    def download(
        cls,
        target_path,
        start_time_unix,
        end_time_unix=None,
        area=None,
        pressure_levels_hpa=None,
        overwrite=False,
    ):
        path = ensure_era5_pressure_level_file(
            target_path=target_path,
            start_time_unix=start_time_unix,
            end_time_unix=end_time_unix,
            area=area,
            pressure_levels_hpa=pressure_levels_hpa,
            overwrite=overwrite,
        )
        return cls.open(path)

    def _longitude_for_dataset(self, lon_deg):
        return _normalise_longitude(self.dataset, lon_deg)

    def _vertical_coord_name(self):
        for name in ("pressure_level", "level"):
            if name in self.dataset.dims or name in self.dataset.coords:
                return name
        raise KeyError("Could not find pressure-level coordinate in ERA5 dataset")

    def _time_coord_name(self):
        return _time_coord_name(self.dataset)

    def _interp_profile(self, lat_deg, lon_deg, unix_time=None):
        ds = self.dataset
        ds = self.dataset[["u", "v", "z"]]
        return _interp_dataset_point(ds, lat_deg, lon_deg, unix_time=unix_time)

    def wind_enu(self, lat_deg, lon_deg, hgt_m, unix_time=None):
        """
        Interpolate ERA5 horizontal wind to a geodetic point.

        Returns eastward, northward, upward velocity in m/s. ERA5 pressure-level
        products do not provide vertical wind here, so the up component is zero.
        """
        hgt_sorted, u_sorted, v_sorted = self.profile_enu(
            lat_deg,
            lon_deg,
            unix_time=unix_time,
        )
        if hgt_sorted.size == 0:
            raise ValueError("ERA5 vertical profile is empty")

        hgt_m = float(hgt_m)
        if hgt_m > float(hgt_sorted[-1]):
            return 0.0, 0.0, 0.0

        east_mps = np.interp(hgt_m, hgt_sorted, u_sorted, left=u_sorted[0], right=u_sorted[-1])
        north_mps = np.interp(hgt_m, hgt_sorted, v_sorted, left=v_sorted[0], right=v_sorted[-1])
        return float(east_mps), float(north_mps), 0.0

    def profile_enu(self, lat_deg, lon_deg, unix_time=None):
        """
        Return an interpolated ERA5 wind profile at one latitude/longitude/time.

        Returns
        -------
        hgt_m : ndarray
            Geometric height in meters.
        east_mps : ndarray
            Eastward wind component in m/s.
        north_mps : ndarray
            Northward wind component in m/s.
        """
        profile = self._interp_profile(lat_deg, lon_deg, unix_time=unix_time)
        self._vertical_coord_name()

        hgt_profile_m = geopotential_to_geometric_height(profile["z"].values)
        u_profile = np.asarray(profile["u"].values, dtype=float)
        v_profile = np.asarray(profile["v"].values, dtype=float)

        order = np.argsort(hgt_profile_m)
        hgt_sorted = np.asarray(hgt_profile_m[order], dtype=float)
        u_sorted = u_profile[order]
        v_sorted = v_profile[order]

        hgt_sorted, unique_idx = np.unique(hgt_sorted, return_index=True)
        u_sorted = u_sorted[unique_idx]
        v_sorted = v_sorted[unique_idx]
        return hgt_sorted, u_sorted, v_sorted

    def wind_ecef(self, lat_deg, lon_deg, hgt_m, unix_time=None):
        east_mps, north_mps, up_mps = self.wind_enu(lat_deg, lon_deg, hgt_m, unix_time=unix_time)
        vec = enu_to_ecef_vector(east_mps, north_mps, up_mps, lat_deg, lon_deg)
        return np.asarray(vec, dtype=float)

    def atmosphere_velocity_eci(self, lat_deg, lon_deg, hgt_m, unix_time, omega=OMEGA_EARTH):
        """
        Inertial atmosphere velocity at a geodetic point.

        This combines Earth co-rotation with the interpolated ERA5 horizontal
        wind field, returning a 3-vector in ECI coordinates.
        """
        pos_ecef = geodetic_to_ecef(lat_deg, lon_deg, hgt_m)
        wind_ecef = self.wind_ecef(lat_deg, lon_deg, hgt_m, unix_time=unix_time)
        omega_vec = np.array([0.0, 0.0, float(omega)])
        vel_ecef = wind_ecef + np.cross(omega_vec, pos_ecef)
        return ecef_to_eci_vector(vel_ecef, unix_time)


def load_or_download_era5(
    target_path,
    start_time_unix,
    end_time_unix=None,
    area=None,
    pressure_levels_hpa=None,
    overwrite=False,
):
    return ERA5Wind.download(
        target_path=target_path,
        start_time_unix=start_time_unix,
        end_time_unix=end_time_unix,
        area=area,
        pressure_levels_hpa=pressure_levels_hpa,
        overwrite=overwrite,
    )


@dataclass
class ERA5ModelLevelWind:
    wind_dataset: xr.Dataset
    surface_dataset: xr.Dataset
    pv: np.ndarray

    def __post_init__(self):
        self.wind_dataset = _sort_dataset_coords(
            self.wind_dataset,
            ("time", "valid_time", "latitude", "longitude", "hybrid"),
        )
        self.surface_dataset = _sort_dataset_coords(
            self.surface_dataset,
            ("time", "valid_time", "latitude", "longitude"),
        )
        self.pv = np.asarray(self.pv, dtype=float)

    @classmethod
    def open(cls, tquv_path, zlnsp_path):
        wind_dataset = xr.open_dataset(tquv_path, engine="cfgrib").load()
        surface_dataset = xr.open_dataset(zlnsp_path, engine="cfgrib").load()
        pv = _get_model_level_pv(tquv_path)
        return cls(wind_dataset=wind_dataset, surface_dataset=surface_dataset, pv=pv)

    @classmethod
    def download(
        cls,
        target_prefix,
        start_time_unix,
        end_time_unix=None,
        area=None,
        model_levels=DEFAULT_MODEL_LEVELS,
        overwrite=False,
    ):
        tquv_path, zlnsp_path = ensure_era5_model_level_files(
            target_prefix=target_prefix,
            start_time_unix=start_time_unix,
            end_time_unix=end_time_unix,
            area=area,
            model_levels=model_levels,
            overwrite=overwrite,
        )
        return cls.open(tquv_path, zlnsp_path)

    def profile_enu(self, lat_deg, lon_deg, unix_time=None):
        wind_profile = _interp_dataset_point(
            self.wind_dataset[["t", "q", "u", "v"]],
            lat_deg,
            lon_deg,
            unix_time=unix_time,
        )
        surface_profile = _interp_dataset_point(
            self.surface_dataset[["z", "lnsp"]],
            lat_deg,
            lon_deg,
            unix_time=unix_time,
        )

        geopotential_ml = _compute_model_level_geopotential_profiles(
            t_profile=wind_profile["t"].values,
            q_profile=wind_profile["q"].values,
            surface_geopotential=surface_profile["z"].values,
            lnsp=surface_profile["lnsp"].values,
            pv=self.pv,
        )
        hgt_profile_m = geopotential_to_geometric_height(geopotential_ml)

        u_profile = np.asarray(wind_profile["u"].values, dtype=float)
        v_profile = np.asarray(wind_profile["v"].values, dtype=float)
        order = np.argsort(hgt_profile_m)
        hgt_sorted = np.asarray(hgt_profile_m[order], dtype=float)
        u_sorted = u_profile[order]
        v_sorted = v_profile[order]
        hgt_sorted, unique_idx = np.unique(hgt_sorted, return_index=True)
        u_sorted = u_sorted[unique_idx]
        v_sorted = v_sorted[unique_idx]
        return hgt_sorted, u_sorted, v_sorted

    def wind_enu(self, lat_deg, lon_deg, hgt_m, unix_time=None):
        hgt_sorted, u_sorted, v_sorted = self.profile_enu(
            lat_deg,
            lon_deg,
            unix_time=unix_time,
        )
        if hgt_sorted.size == 0:
            raise ValueError("ERA5 model-level vertical profile is empty")
        east_mps = np.interp(float(hgt_m), hgt_sorted, u_sorted, left=u_sorted[0], right=u_sorted[-1])
        north_mps = np.interp(float(hgt_m), hgt_sorted, v_sorted, left=v_sorted[0], right=v_sorted[-1])
        return float(east_mps), float(north_mps), 0.0

    def wind_ecef(self, lat_deg, lon_deg, hgt_m, unix_time=None):
        east_mps, north_mps, up_mps = self.wind_enu(lat_deg, lon_deg, hgt_m, unix_time=unix_time)
        return np.asarray(
            enu_to_ecef_vector(east_mps, north_mps, up_mps, lat_deg, lon_deg),
            dtype=float,
        )

    def atmosphere_velocity_eci(self, lat_deg, lon_deg, hgt_m, unix_time, omega=OMEGA_EARTH):
        pos_ecef = geodetic_to_ecef(lat_deg, lon_deg, hgt_m)
        wind_ecef = self.wind_ecef(lat_deg, lon_deg, hgt_m, unix_time=unix_time)
        omega_vec = np.array([0.0, 0.0, float(omega)])
        vel_ecef = wind_ecef + np.cross(omega_vec, pos_ecef)
        return ecef_to_eci_vector(vel_ecef, unix_time)


def load_or_download_era5_model_levels(
    target_prefix,
    start_time_unix,
    end_time_unix=None,
    area=None,
    model_levels=DEFAULT_MODEL_LEVELS,
    overwrite=False,
):
    return ERA5ModelLevelWind.download(
        target_prefix=target_prefix,
        start_time_unix=start_time_unix,
        end_time_unix=end_time_unix,
        area=area,
        model_levels=model_levels,
        overwrite=overwrite,
    )

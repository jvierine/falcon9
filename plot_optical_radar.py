import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as n

PANEL_CUSTOM_OFFSETS = {
    "1": [2, 0],
    "2": [-2, 0],
    "3": [4, 0],
    "4": [-2.5, 0],
    "5": [3, 0],
    "8": [-3, 0],
    "7": [-6, 0],
    "9": [4, -0.3],
    "c": [-4, -0.2],
    "a": [-5, 0],
    "h": [-6, 0],
    "g": [-7, 0],
    "o": [3, 0],
    "n": [5, 0],
    "p": [4, 0.2],
    "r": [8, 0.4],
    "s": [4, 0.0],
    "w": [6, 0.0],
    "v": [7, 0.3],
    "x": [2, 0.0],
    "z": [2, 0.2],
    "t": [-6, -0.2],
    "u": [-6, -0.2],
    "e": [-7, -0.2],
    "d": [-7, -0.2],
    "i": [-6, -0.2],
    "m": [-6, -0.2],
    "l": [-8, -0.2],
    "j": [-10, -0.2],
    "k": [-12, -0.2],
}


def publication_panel_style(font_scale=1.0):
    scale = float(font_scale)
    return {
        "axis_label_fontsize": 17.0 * scale,
        "tick_label_fontsize": 14.0 * scale,
        "legend_fontsize": 12.0 * scale,
        "annotation_fontsize": 8.0 * scale,
        "colorbar_label_fontsize": 16.0 * scale,
        "title_fontsize": 15.0 * scale,
        "line_width": 1.1,
        "marker_size": 4.0,
    }


def publication_rcparams(font_scale=1.0):
    style = publication_panel_style(font_scale=font_scale)
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": style["tick_label_fontsize"],
        "axes.titlesize": style["title_fontsize"],
        "axes.labelsize": style["axis_label_fontsize"],
        "legend.fontsize": style["legend_fontsize"],
        "xtick.labelsize": style["tick_label_fontsize"],
        "ytick.labelsize": style["tick_label_fontsize"],
        "lines.linewidth": style["line_width"],
        "lines.markersize": style["marker_size"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }


def _panel_fragment_data():
    import plot_fragments as plf

    return plf.get_fragments()


def _panel_radar_data():
    import plot_fragments as plf

    return plf.get_radar_detections()


def _plot_optical_measurements(
    ax,
    fragment_ids,
    fragment_geo_pos,
    fragment_pos_err,
    optical_color_mode="gray",
    show_ids=True,
    annotation_fontsize=8.0,
):
    meters_to_deg_lat = 1.0 / 111320.0

    for i in range(len(fragment_ids)):
        lon_pts = fragment_geo_pos[i][:, 1]
        lat_pts = fragment_geo_pos[i][:, 0]
        alt_pts_km = fragment_geo_pos[i][:, 2] / 1e3
        err_m = 2.0 * fragment_pos_err[i]
        lon_err_deg = err_m * (meters_to_deg_lat / np.cos(np.deg2rad(lat_pts)))

        point_color = "gray" if optical_color_mode == "gray" else f"C{i % 10}"
        edge_color = "lightgray" if optical_color_mode == "gray" else "gray"

        ax.errorbar(
            lon_pts,
            alt_pts_km,
            xerr=lon_err_deg,
            yerr=(err_m / 1e3),
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=15,
            ecolor=edge_color,
            elinewidth=0.8,
            capsize=2,
            color=point_color,
            alpha=0.9,
        )

        if not show_ids:
            continue

        fid = fragment_ids[i]
        loni0 = np.argmin(lon_pts)
        alt0 = alt_pts_km[loni0]
        lon0 = lon_pts[loni0]
        offset = PANEL_CUSTOM_OFFSETS.get(fid, [2, 0])
        label_color = "gray" if optical_color_mode == "gray" else point_color
        if fid not in ("1", "2"):
            ax.scatter(lon0 + offset[1], alt0 + offset[0], s=120, color="white", zorder=19)
            ax.text(
                lon0 + offset[1],
                alt0 + offset[0],
                fid,
                color=label_color,
                fontsize=annotation_fontsize,
                va="center",
                ha="center",
                zorder=20,
            )


def _plot_optical_measurements_time(
    ax,
    fragment_ids,
    fragment_geo_pos,
    fragment_pos_err,
    fragment_times,
    optical_color_mode="gray",
):
    import plot_fragments as plf

    for i in range(len(fragment_ids)):
        tvals = plf.unix_to_datetime(fragment_times[i])
        alt_pts_km = fragment_geo_pos[i][:, 2] / 1e3
        err_m = 2.0 * fragment_pos_err[i]

        point_color = "gray" if optical_color_mode == "gray" else f"C{i % 10}"
        edge_color = "lightgray" if optical_color_mode == "gray" else "gray"

        ax.errorbar(
            tvals,
            alt_pts_km,
            yerr=(err_m / 1e3),
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=15,
            ecolor=edge_color,
            elinewidth=0.8,
            capsize=2,
            color=point_color,
            alpha=0.9,
        )


def _format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))


def _annotate_orbgen_fragment_columns(
    ax,
    fragment_ids,
    fragment_geo_pos,
    fragment_times,
    annotation_fontsize=12.0,
    altitude_margin_km=7.0,
    stack_spacing_km=3.8,
):
    columns = {}

    for i, fid in enumerate(fragment_ids):
        lon_pts = np.asarray(fragment_geo_pos[i][:, 1], dtype=float)
        alt_pts_km = np.asarray(fragment_geo_pos[i][:, 2], dtype=float) / 1e3
        times = np.asarray(fragment_times[i])
        if lon_pts.size == 0 or alt_pts_km.size == 0 or times.size == 0:
            continue

        start_idx = int(np.argmin(times))
        start_lon = float(lon_pts[start_idx])
        column_lon = float(np.round(start_lon))

        columns.setdefault(column_lon, []).append(
            {
                "fid": fid,
                "lowest_alt": float(np.min(alt_pts_km)),
                "color": f"C{i % 10}",
            }
        )

    for column_lon, items in columns.items():
        items.sort(key=lambda item: (-item["lowest_alt"], item["fid"]))
        nearby_alts = []
        for geo_pos in fragment_geo_pos:
            lon_pts = np.asarray(geo_pos[:, 1], dtype=float)
            alt_pts_km = np.asarray(geo_pos[:, 2], dtype=float) / 1e3
            mask = np.isfinite(lon_pts) & np.isfinite(alt_pts_km) & (np.abs(lon_pts - column_lon) <= 1.0)
            if np.any(mask):
                nearby_alts.append(float(np.min(alt_pts_km[mask])))

        if len(nearby_alts) > 0:
            base_alt = min(nearby_alts) - altitude_margin_km
        else:
            base_alt = min(item["lowest_alt"] for item in items) - altitude_margin_km

        for idx, item in enumerate(items):
            label_alt = base_alt - idx * stack_spacing_km
            ax.text(
                column_lon,
                label_alt,
                item["fid"],
                color=item["color"],
                fontsize=annotation_fontsize,
                fontweight="bold",
                va="top",
                ha="center",
                zorder=30,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 0.35,
                },
            )


def _annotate_orbgen_fragment_time_bins(
    ax,
    fragment_ids,
    fragment_geo_pos,
    fragment_times,
    annotation_fontsize=12.0,
    time_bin_seconds=20.0,
    altitude_margin_km=0,
    stack_spacing_km=2.8,
):
    import plot_fragments as plf

    bins = {}

    for i, fid in enumerate(fragment_ids):
        alt_pts_km = np.asarray(fragment_geo_pos[i][:, 2], dtype=float) / 1e3
        times = np.asarray(fragment_times[i], dtype=float)
        if alt_pts_km.size == 0 or times.size == 0:
            continue

        start_idx = int(np.argmin(times))
        start_time = float(times[start_idx])
        bin_start = time_bin_seconds * np.floor(start_time / time_bin_seconds)
        bin_center = bin_start + 0.5 * time_bin_seconds

        bins.setdefault(bin_center, []).append(
            {
                "fid": fid,
                "lowest_alt": float(np.min(alt_pts_km)),
                "color": f"C{i % 10}",
            }
        )

    for bin_center, items in bins.items():
        items.sort(key=lambda item: (-item["lowest_alt"], item["fid"]))
        nearby_alts = []
        for geo_pos, times in zip(fragment_geo_pos, fragment_times):
            alt_pts_km = np.asarray(geo_pos[:, 2], dtype=float) / 1e3
            times = np.asarray(times, dtype=float)
            mask = np.isfinite(times) & np.isfinite(alt_pts_km) & (np.abs(times - bin_center) <= time_bin_seconds)
            if np.any(mask):
                nearby_alts.append(float(np.min(alt_pts_km[mask])))

        if len(nearby_alts) > 0:
            base_alt = min(nearby_alts) - altitude_margin_km
        else:
            base_alt = min(item["lowest_alt"] for item in items) - altitude_margin_km

        label_time = plf.unix_to_datetime([bin_center])[0]
        for idx, item in enumerate(items):
            label_alt = base_alt - idx * stack_spacing_km
            ax.text(
                label_time,
                label_alt,
                item["fid"],
                color=item["color"],
                fontsize=annotation_fontsize,
                fontweight="bold",
                va="top",
                ha="center",
                zorder=30,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 0.35,
                },
            )


def plot_orbgen_lon_alt_panel(ax, title=None, font_scale=1.0):
    from matplotlib.lines import Line2D

    style = publication_panel_style(font_scale=font_scale)
    _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = _panel_fragment_data()

    _plot_optical_measurements(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_pos_err,
        optical_color_mode="color",
        show_ids=False,
        annotation_fontsize=style["annotation_fontsize"],
    )

    _annotate_orbgen_fragment_columns(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_times,
        annotation_fontsize=max(12.0, style["annotation_fontsize"] * 1.5),
    )

    for info in frags.values():
        ax.axvline(
            info["lon"],
            color="red",
            linestyle="--",
            linewidth=0.8,
            zorder=8,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=0.8))
    labels.append("Ground recovered fragment")
    ax.legend(handles=handles, labels=labels, frameon=False, loc="lower left", fontsize=style["legend_fontsize"])
    ax.tick_params(labelsize=style["tick_label_fontsize"])
    ax.set_xlabel("Longitude (deg)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Altitude (km)", fontsize=style["axis_label_fontsize"])
    if title is not None:
        ax.set_title(title, fontsize=style["title_fontsize"])
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0.0)
    return ax


def plot_orbgen_time_alt_panel(ax, title=None, font_scale=1.0):
    style = publication_panel_style(font_scale=font_scale)
    _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = _panel_fragment_data()

    _plot_optical_measurements_time(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_pos_err,
        fragment_times,
        optical_color_mode="color",
    )

    _annotate_orbgen_fragment_time_bins(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_times,
        annotation_fontsize=max(12.0, style["annotation_fontsize"] * 1.5),
    )

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles=handles, labels=labels, frameon=False, loc="lower left", fontsize=style["legend_fontsize"])
    ax.tick_params(labelsize=style["tick_label_fontsize"])
    ax.set_xlabel("Time (UTC)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Altitude (km)", fontsize=style["axis_label_fontsize"])
    if title is not None:
        ax.set_title(title, fontsize=style["title_fontsize"])
    _format_time_axis(ax)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0.0)
    return ax


def plot_radar_lon_alt_panel(ax, title=None, font_scale=1.0):
    style = publication_panel_style(font_scale=font_scale)
    _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = _panel_fragment_data()
    rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop, txid, rxid = _panel_radar_data()

    _plot_optical_measurements(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_pos_err,
        optical_color_mode="gray",
        show_ids=False,
        annotation_fontsize=style["annotation_fontsize"],
    )

    for i in range(len(rlat)):
        idx = n.where(rsnr[i] > -10)[0]
        if len(idx) > 0:
            ax.plot(
                rlon[i][idx],
                ralt[i][idx] / 1e3,
                ".",
                color=f"C{i % 10}",
                ms=5,
                zorder=999,
                label=f"{txid[i]}-{rxid[i]}",
            )

    ax.legend(frameon=False, loc="lower left", fontsize=style["legend_fontsize"])
    ax.tick_params(labelsize=style["tick_label_fontsize"])
    ax.set_xlabel("Longitude (deg)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Altitude (km)", fontsize=style["axis_label_fontsize"])
    if title is not None:
        ax.set_title(title, fontsize=style["title_fontsize"])
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlim([3, 17.5])
    ax.set_ylim(bottom=0.0)
    return ax


def plot_radar_time_alt_panel(ax, title=None, font_scale=1.0):
    import plot_fragments as plf

    style = publication_panel_style(font_scale=font_scale)
    _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = _panel_fragment_data()
    rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop, txid, rxid = _panel_radar_data()

    _plot_optical_measurements_time(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_pos_err,
        fragment_times,
        optical_color_mode="gray",
    )

    for i in range(len(rlat)):
        idx = n.where(rsnr[i] > -10)[0]
        if len(idx) > 0:
            ax.plot(
                plf.unix_to_datetime(rtime[i][idx]),
                ralt[i][idx] / 1e3,
                ".",
                color=f"C{i % 10}",
                ms=5,
                zorder=999,
                label=f"{txid[i]}-{rxid[i]}",
            )

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles=handles, labels=labels, frameon=False, loc="lower left", fontsize=style["legend_fontsize"])
    ax.tick_params(labelsize=style["tick_label_fontsize"])
    ax.set_xlabel("Time (UTC)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Altitude (km)", fontsize=style["axis_label_fontsize"])
    if title is not None:
        ax.set_title(title, fontsize=style["title_fontsize"])
    _format_time_axis(ax)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0.0)
    return ax


def _load_fit_overlay_data(field_name, positive_only=False):
    import h5py
    import matplotlib.colors as mcolors
    from pathlib import Path

    ballistic_fit_files = sorted(Path(__file__).parent.glob("ballistic_fit_sharedstart*.h5"))

    def load_segment(group):
        order = np.argsort(group["times_model"][()])
        return {
            "times_model": group["times_model"][()][order],
            "lon_deg": group["lon_deg"][()][order],
            "hgt_m": group["hgt_m"][()][order],
            field_name: group[field_name][()][order],
        }

    def load_fit_result(path):
        with h5py.File(path, "r") as h5:
            result = {
                "label": Path(path).stem,
                "model": load_segment(h5["model"]),
                "impact": None,
            }
            if "impact" in h5 and "trajectory" in h5["impact"]:
                result["impact"] = {"trajectory": load_segment(h5["impact/trajectory"])}
        return result

    fit_results = [load_fit_result(path) for path in ballistic_fit_files]
    if len(fit_results) == 0:
        raise FileNotFoundError("No ballistic_fit_sharedstart*.h5 files were found.")

    valid_values = []
    for result in fit_results:
        for segment in [result["model"]] + ([result["impact"]["trajectory"]] if result["impact"] is not None else []):
            values = np.asarray(segment[field_name], dtype=float)
            mask = np.isfinite(values)
            if positive_only:
                mask &= values > 0.0
            if np.any(mask):
                valid_values.append(values[mask])

    if len(valid_values) == 0:
        raise ValueError(f"No valid values were found for {field_name} in the ballistic fits.")

    all_values = np.concatenate(valid_values)
    return fit_results, float(np.nanmin(all_values)), float(np.nanmax(all_values))


def plot_fit_overlay_lon_alt_panel(
    ax,
    field_name,
    colorbar_label,
    cmap_name,
    log_color=False,
    positive_only=False,
    title=None,
    font_scale=1.0,
    add_colorbar=True,
):
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    style = publication_panel_style(font_scale=font_scale)
    _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = _panel_fragment_data()
    fit_results, vmin, vmax = _load_fit_overlay_data(field_name, positive_only=positive_only)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_color else mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    def add_colored_trajectory(segment, linewidth, alpha, linestyle="solid", zorder=35):
        lon = np.asarray(segment["lon_deg"], dtype=float)
        hgt_km = np.asarray(segment["hgt_m"], dtype=float) / 1e3
        values = np.asarray(segment[field_name], dtype=float)
        mask = np.isfinite(lon) & np.isfinite(hgt_km)
        if np.count_nonzero(mask) < 2:
            return None

        lon = lon[mask]
        hgt_km = hgt_km[mask]
        values = values[mask]

        if positive_only:
            values = np.where(np.isfinite(values) & (values > 0.0), values, float(norm.vmin))
        else:
            finite = np.isfinite(values)
            if not np.any(finite):
                return None
            fill_value = float(np.nanmin(values[finite]))
            values = np.where(finite, values, fill_value)

        points = np.column_stack((lon, hgt_km))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        seg_values = np.sqrt(values[:-1] * values[1:]) if log_color else 0.5 * (values[:-1] + values[1:])
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=linewidth,
            alpha=alpha,
            linestyles=linestyle,
            zorder=zorder,
        )
        lc.set_array(seg_values)
        ax.add_collection(lc)
        return lc

    def add_outline_trajectory(segment):
        lon = np.asarray(segment["lon_deg"], dtype=float)
        hgt_km = np.asarray(segment["hgt_m"], dtype=float) / 1e3
        mask = np.isfinite(lon) & np.isfinite(hgt_km)
        if np.count_nonzero(mask) < 2:
            return None
        return ax.plot(
            lon[mask],
            hgt_km[mask],
            color="black",
            linewidth=0.7,
            alpha=0.9,
            linestyle="dashed",
            zorder=38,
        )[0]

    _plot_optical_measurements(
        ax,
        fragment_ids,
        fragment_geo_pos,
        fragment_pos_err,
        optical_color_mode="gray",
        show_ids=False,
        annotation_fontsize=style["annotation_fontsize"],
    )

    for info in frags.values():
        ax.axvline(
            info["lon"],
            color="red",
            linestyle="--",
            alpha=0.2,
            linewidth=0.8,
            zorder=-8,
        )

    color_handle = None
    for result in fit_results:
        handle = add_colored_trajectory(result["model"], linewidth=2.4, alpha=0.95, linestyle="solid", zorder=36)
        if color_handle is None and handle is not None:
            color_handle = handle
        if result["impact"] is not None:
            impact_handle = add_colored_trajectory(
                result["impact"]["trajectory"],
                linewidth=2.0,
                alpha=0.9,
                linestyle="dashed",
                zorder=37,
            )
            if color_handle is None and impact_handle is not None:
                color_handle = impact_handle
            add_outline_trajectory(result["impact"]["trajectory"])

    if add_colorbar and color_handle is not None:
        cbar = ax.figure.colorbar(color_handle, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=style["colorbar_label_fontsize"])
        cbar.ax.tick_params(labelsize=style["tick_label_fontsize"])

    legend_handles = [
        Line2D([], [], linestyle="None", marker=".", color="gray", markersize=6, label="Optical detection"),
        Line2D([], [], color="black", linewidth=2.2, label="Ballistic fit"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Extrapolated trajectory"),
        Line2D([], [], color="red", linestyle="--", linewidth=0.8, label="Ground recovered fragment"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=style["legend_fontsize"])
    ax.tick_params(labelsize=style["tick_label_fontsize"])
    ax.set_xlabel("Longitude (deg)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Altitude (km)", fontsize=style["axis_label_fontsize"])
    if title is not None:
        ax.set_title(title, fontsize=style["title_fontsize"])
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlim([-2.5, 20.5])
    ax.set_ylim(bottom=0.0)
    return ax


def plot_fit_energy_lon_alt_panel(ax, title=None, font_scale=1.0, add_colorbar=True):
    return plot_fit_overlay_lon_alt_panel(
        ax,
        field_name="specific_energy_loss_rate_w_kg",
        colorbar_label=r"Specific kinetic energy loss rate (W kg$^{-1}$)",
        cmap_name="inferno",
        log_color=True,
        positive_only=True,
        title=title,
        font_scale=font_scale,
        add_colorbar=add_colorbar,
    )


def plot_fit_relative_speed_lon_alt_panel(ax, title=None, font_scale=1.0, add_colorbar=True):
    return plot_fit_overlay_lon_alt_panel(
        ax,
        field_name="relative_speed_m_s",
        colorbar_label=r"Velocity relative to atmosphere (m s$^{-1}$)",
        cmap_name="viridis",
        log_color=False,
        positive_only=False,
        title=title,
        font_scale=font_scale,
        add_colorbar=add_colorbar,
    )

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


def optical_plot():
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
    axis_label_fontsize = 18
    tick_label_fontsize = 15
    legend_fontsize = 13
    annotation_fontsize = 8
    colorbar_label_fontsize = 17

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
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



def radar_plot():
    import os
    import plot_fragments as plf
    from matplotlib.lines import Line2D

    hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()
    rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop,txid,rxid = plf.get_radar_detections()

    orb = OrbGenParser("data/orbgen#12-cut.dat")

    start_time = "2025-02-19T03:42:45"
    stop_time = "2025-02-19T03:50:00"

    # Use the dataframe directly
    df = orb.df

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
    fn_lon_alt = os.path.join(out_dir, "opt_radar_lon_vs_alt.pdf")

    # helper: meters -> degrees conversion for latitude and longitude
    meters_to_deg_lat = 1.0 / 111320.0  # approx

    # Prepare radar arrays as numpy
    #rlat = np.asarray(rlat)
    #rlon = np.asarray(rlon)
    #ralt = np.asarray(ralt)
    #rsnr = np.asarray(rsnr)


    # Plot 1: lon vs alt (ground track) with optical detections in gray
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    custom_offsets = {
        "1":[2,0], "2":[-2,0], "3":[4,0], "4":[-2.5,0], "5":[3,0], "8":[-3,0],
        "7":[-6,0], "9":[4,-0.3], "c":[-4,-0.2], "a":[-5,0], "h":[-6,0], "g":[-7,0],
        "o":[3,0], "n":[5,0], "p":[4,0.2], "r":[8,0.4], "s":[4,0.0], "w":[6,0.0],
        "v":[7,0.3], "x":[2,0.0], "z":[2,0.2], "t":[-6,-0.2], "u":[-6,-0.2],
        "e":[-7,-0.2], "d":[-7,-0.2], "i":[-6,-0.2], "m":[-6,-0.2], "l":[-8,-0.2],
        "j":[-10,-0.2], "k":[-12,-0.2],
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

        # optical detections in gray
        ax1.errorbar(
            lon_pts,
            alt_pts_km,
            xerr=lon_err_deg,
            yerr=(err_m / 1e3),
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=15,
            ecolor="lightgray",
            elinewidth=0.8,
            capsize=2,
            color="gray"
        )
        fid = fragment_ids[i]
        loni0 = np.argmin(lon_pts)
        alt0 = alt_pts_km[loni0]
        lon0 = lon_pts[loni0]
        if fid in custom_offsets.keys():
            offset = custom_offsets[fid]
        else:
            offset = [2, 0]
        # label in gray for consistency
        if fid != '1' and fid != '2':
            ax1.scatter(lon0 + offset[1], alt0 + offset[0], s=120, color="white", zorder=19)
            ax1.text(
                lon0 + offset[1],
                alt0 + offset[0],
                fid,
                color="gray",
                fontsize=8,
                va="center",
                ha="center",
                zorder=20
            )

    for i in range(len(rlat)):
        idx=n.where(rsnr[i]>-10)[0] 
        if len(idx)>0:           
            # Overlay radar detections colored by SNR
            sc1 = ax1.plot(
                rlon[i][idx],
                ralt[i][idx]/1e3,
                ".",
                color="C%d"%(i),
                ms=5,
                zorder=999,
                label="%s-%s"%(txid[i],rxid[i])
            )
#    cbar1 = fig1.colorbar(sc1, ax=ax1, pad=0.02)
#    cbar1.set_label("Radar SNR (dB)")
    ax1.legend()
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Radar detections and optical fragment detections")
    ax1.grid(True, linestyle="--", linewidth=0.5)
    # single legend entry for optical + radar
    ax1.legend(frameon=False)
    ax1.set_xlim([3,17.5])
    fig1.tight_layout()
    fig1.savefig(fn_lon_alt, bbox_inches="tight")
    plt.close(fig1)

    print(f"Saved radar overlay PDF: {fn_lon_alt}")


def _fit_overlay_plot(
    field_name,
    output_filename,
    colorbar_label,
    title,
    cmap_name,
    log_color=False,
    positive_only=False,
):
    import os
    from pathlib import Path

    import h5py
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    import plot_fragments as plf
    from matplotlib.lines import Line2D

    ballistic_fit_files = sorted(Path(__file__).parent.glob("ballistic_fit_sharedstart*.h5"))
    #print(sharedstart_files)

    def load_segment(group):
        order = np.argsort(group["times_model"][()])
        return {
            "times_model": group["times_model"][()][order],
            "lon_deg": group["lon_deg"][()][order],
            "hgt_m": group["hgt_m"][()][order],
            field_name: group[field_name][()][order],
        }

    def load_fit_result(path):
        with h5py.File(path, "r") as h5:
            result = {
                "label": Path(path).stem,
                "model": load_segment(h5["model"]),
                "impact": None,
            }
            if "impact" in h5 and "trajectory" in h5["impact"]:
                result["impact"] = {
                    "trajectory": load_segment(h5["impact/trajectory"]),
                }
        return result

    def add_colored_trajectory(ax, segment, cmap, norm, linewidth, alpha, linestyle="solid", zorder=35):
        lon = np.asarray(segment["lon_deg"], dtype=float)
        hgt_km = np.asarray(segment["hgt_m"], dtype=float) / 1e3
        values = np.asarray(segment[field_name], dtype=float)
        mask = np.isfinite(lon) & np.isfinite(hgt_km)
        if np.count_nonzero(mask) < 2:
            return None

        lon = lon[mask]
        hgt_km = hgt_km[mask]
        values = values[mask]

        if positive_only:
            values = np.where(np.isfinite(values) & (values > 0.0), values, float(norm.vmin))
        else:
            finite = np.isfinite(values)
            if not np.any(finite):
                return None
            fill_value = float(np.nanmin(values[finite]))
            values = np.where(finite, values, fill_value)

        points = np.column_stack((lon, hgt_km))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        if log_color:
            seg_values = np.sqrt(values[:-1] * values[1:])
        else:
            seg_values = 0.5 * (values[:-1] + values[1:])

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=linewidth,
            alpha=alpha,
            linestyles=linestyle,
            zorder=zorder,
        )
        lc.set_array(seg_values)
        ax.add_collection(lc)
        return lc

    def add_outline_trajectory(
        ax,
        segment,
        color="black",
        linewidth=0.8,
        alpha=0.95,
        linestyle="dashed",
        zorder=38,
    ):
        lon = np.asarray(segment["lon_deg"], dtype=float)
        hgt_km = np.asarray(segment["hgt_m"], dtype=float) / 1e3
        mask = np.isfinite(lon) & np.isfinite(hgt_km)
        if np.count_nonzero(mask) < 2:
            return None

        return ax.plot(
            lon[mask],
            hgt_km[mask],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )[0]

    hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()

    fit_results = []
    for file_name in ballistic_fit_files:
        #fit_path = Path(__file__).with_name(file_name)
        #if fit_path.exists():
        fit_results.append(load_fit_result(file_name))
#        else:
 #           print(f"Skipping missing fit file: {fit_path}")

    if len(fit_results) == 0:
        raise FileNotFoundError("No ballistic_fit_*.h5 files were found for fit_overlay_plot().")

    valid_values = []
    for result in fit_results:
        for segment in [result["model"]] + ([result["impact"]["trajectory"]] if result["impact"] is not None else []):
            values = np.asarray(segment[field_name], dtype=float)
            mask = np.isfinite(values)
            if positive_only:
                mask &= values > 0.0
            if np.any(mask):
                valid_values.append(values[mask])

    if len(valid_values) == 0:
        raise ValueError(f"No valid values were found for {field_name} in the ballistic fits.")

    all_values = np.concatenate(valid_values)
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if log_color:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    axis_label_fontsize = 18
    tick_label_fontsize = 15
    legend_fontsize = 13
    annotation_fontsize = 8
    colorbar_label_fontsize = 17

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    out_dir = "."
    fn_lon_alt = os.path.join(out_dir, output_filename)
    meters_to_deg_lat = 1.0 / 111320.0

    fig1, ax1 = plt.subplots(1, 1, figsize=(8.2, 5.8))
    custom_offsets = {
        "1": [2, 0], "2": [-2, 0], "3": [4, 0], "4": [-2.5, 0], "5": [3, 0], "8": [-3, 0],
        "7": [-6, 0], "9": [4, -0.3], "c": [-4, -0.2], "a": [-5, 0], "h": [-6, 0], "g": [-7, 0],
        "o": [3, 0], "n": [5, 0], "p": [4, 0.2], "r": [8, 0.4], "s": [4, 0.0], "w": [6, 0.0],
        "v": [7, 0.3], "x": [2, 0.0], "z": [2, 0.2], "t": [-6, -0.2], "u": [-6, -0.2],
        "e": [-7, -0.2], "d": [-7, -0.2], "i": [-6, -0.2], "m": [-6, -0.2], "l": [-8, -0.2],
        "j": [-10, -0.2], "k": [-12, -0.2],
    }

    for i in range(len(fragment_ids)):
        lon_pts = fragment_geo_pos[i][:, 1]
        lat_pts = fragment_geo_pos[i][:, 0]
        alt_pts_km = fragment_geo_pos[i][:, 2] / 1e3
        err_m = 2.0 * fragment_pos_err[i]
        lon_err_deg = err_m * (meters_to_deg_lat / np.cos(np.deg2rad(lat_pts)))

        ax1.errorbar(
            lon_pts,
            alt_pts_km,
            xerr=lon_err_deg,
            yerr=(err_m / 1e3),
            fmt=".",
            label="Optical detection" if i == 0 else None,
            zorder=10,
            ecolor="lightgray",
            elinewidth=0.8,
            capsize=2,
            color="gray",
            alpha=0.9,
        )

        fid = fragment_ids[i]
        loni0 = np.argmin(lon_pts)
        alt0 = alt_pts_km[loni0]
        lon0 = lon_pts[loni0]
        offset = custom_offsets.get(fid, [2, 0])
        if fid not in ("1", "2"):
            ax1.scatter(lon0 + offset[1], alt0 + offset[0], s=120, color="white", zorder=19)
            ax1.text(
                lon0 + offset[1],
                alt0 + offset[0],
                fid,
                color="gray",
                fontsize=annotation_fontsize,
                va="center",
                ha="center",
                zorder=20,
            )

    for fid, info in frags.items():
        ax1.axvline(
            info["lon"],
            color="red",
            linestyle="--",
            alpha=0.2,
            linewidth=0.8,
            zorder=-8,
        )

    color_handle = None
    for result in fit_results:
        handle = add_colored_trajectory(
            ax1,
            result["model"],
            cmap,
            norm,
            linewidth=2.4,
            alpha=0.95,
            linestyle="solid",
            zorder=36,
        )
        if color_handle is None and handle is not None:
            color_handle = handle
        if result["impact"] is not None:
            impact_handle = add_colored_trajectory(
                ax1,
                result["impact"]["trajectory"],
                cmap,
                norm,
                linewidth=2.0,
                alpha=0.9,
                linestyle="dashed",
                zorder=37,
            )
            if color_handle is None and impact_handle is not None:
                color_handle = impact_handle
            add_outline_trajectory(
                ax1,
                result["impact"]["trajectory"],
                color="black",
                linewidth=0.7,
                alpha=0.9,
                linestyle="dashed",
                zorder=38,
            )

    if color_handle is not None:
        cbar = fig1.colorbar(color_handle, ax=ax1, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)

    legend_handles = [
        Line2D([], [], linestyle="None", marker=".", color="gray", markersize=6, label="Optical detection"),
        Line2D([], [], color="black", linewidth=2.2, label="Ballistic fit"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Extrapolated trajectory"),
        Line2D([], [], color="red", linestyle="--", linewidth=0.8, label="Ground recovered fragment"),
    ]
    ax1.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=legend_fontsize)
    ax1.tick_params(labelsize=tick_label_fontsize)
    ax1.set_xlabel("Longitude (deg)", fontsize=axis_label_fontsize)
    ax1.set_ylabel("Altitude (km)", fontsize=axis_label_fontsize)
    ax1.grid(True, linestyle="--", linewidth=0.5)
    ax1.set_xlim([-2.5, 20.5])
    ax1.set_ylim(bottom=0.0)
    fig1.tight_layout()
    fig1.savefig(fn_lon_alt, bbox_inches="tight")
    plt.close(fig1)

    print(f"Saved fit overlay PDF: {fn_lon_alt}")


def fit_overlay_plot():
    _fit_overlay_plot(
        field_name="specific_energy_loss_rate_w_kg",
        output_filename="opt_fit_lon_vs_alt.pdf",
        colorbar_label=r"Specific kinetic energy loss rate (W kg$^{-1}$)",
        title="Ballistic fits over optical fragment detections",
        cmap_name="inferno",
        log_color=True,
        positive_only=True,
    )


def fit_overlay_velocity_plot():
    _fit_overlay_plot(
        field_name="relative_speed_m_s",
        output_filename="opt_fit_relative_speed_lon_vs_alt.pdf",
        colorbar_label=r"Velocity relative to atmosphere (m s$^{-1}$)",
        title="Ballistic fits colored by atmosphere-relative velocity",
        cmap_name="viridis",
        log_color=False,
        positive_only=False,
    )


def main():
    fit_overlay_plot()
    fit_overlay_velocity_plot()


if __name__ == "__main__":
    main()

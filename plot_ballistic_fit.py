from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as n
from matplotlib.lines import Line2D

from ground_reco import frags


FIT_FILES = [
    Path(__file__).with_name("ballistic_fit_1.h5"),
    Path(__file__).with_name("ballistic_fit_2.h5"),
]

FIT_MARKERS = [
    ("o", "s"),
    ("^", "D"),
]

FIT_COLORS = [
    "tab:blue",
    "tab:orange",
]

RECOVERED_FRAGMENTS = [
    (frag_id, info["lat"], info["lon"])
    for frag_id, info in sorted(frags.items())
]


def load_segment(group):
    order = n.argsort(group["times_model"][()])
    return {
        "times_model": group["times_model"][()][order],
        "lat_deg": group["lat_deg"][()][order],
        "lon_deg": group["lon_deg"][()][order],
        "hgt_m": group["hgt_m"][()][order],
        "rho_a_kg_m3": group["rho_a_kg_m3"][()][order],
        "relative_speed_m_s": group["relative_speed_m_s"][()][order],
        "specific_energy_loss_rate_w_kg": group["specific_energy_loss_rate_w_kg"][()][order],
    }


def load_fit_result(path):
    with h5py.File(path, "r") as h5:
        result = {
            "label": path.stem,
            "model": load_segment(h5["model"]),
            "impact": None,
            "impact_uncertainty": None,
        }
        if "impact" in h5 and "trajectory" in h5["impact"]:
            result["impact"] = {
                "trajectory": load_segment(h5["impact/trajectory"]),
                "impact_lat_deg": float(h5["impact/impact_lat_deg"][()]),
                "impact_lon_deg": float(h5["impact/impact_lon_deg"][()]),
                "impact_hgt_m": float(h5["impact/impact_hgt_m"][()]),
            }
        if "impact_uncertainty" in h5:
            group = h5["impact_uncertainty"]
            if (
                "impact_horizontal_major_axis_1sigma_m" in group
                and "impact_horizontal_minor_axis_1sigma_m" in group
                and "impact_horizontal_major_axis_azimuth_deg" in group
            ):
                result["impact_uncertainty"] = {
                    "impact_horizontal_major_axis_1sigma_m": float(group["impact_horizontal_major_axis_1sigma_m"][()]),
                    "impact_horizontal_minor_axis_1sigma_m": float(group["impact_horizontal_minor_axis_1sigma_m"][()]),
                    "impact_horizontal_major_axis_azimuth_deg": float(group["impact_horizontal_major_axis_azimuth_deg"][()]),
                }
    return result


def build_impact_uncertainty_ellipse_lon_lat(
    impact_lat_deg,
    impact_lon_deg,
    major_axis_m,
    minor_axis_m,
    azimuth_deg,
    n_points=181,
):
    theta = n.linspace(0.0, 2.0 * n.pi, int(n_points), endpoint=True)
    azimuth_rad = n.deg2rad(float(azimuth_deg))

    east = major_axis_m * n.sin(theta) * n.sin(azimuth_rad) + minor_axis_m * n.cos(theta) * n.cos(azimuth_rad)
    north = major_axis_m * n.sin(theta) * n.cos(azimuth_rad) - minor_axis_m * n.cos(theta) * n.sin(azimuth_rad)

    lat_scale_m_per_deg = 111320.0
    lon_scale_m_per_deg = lat_scale_m_per_deg * max(n.cos(n.deg2rad(float(impact_lat_deg))), 1e-6)

    lon = float(impact_lon_deg) + east / lon_scale_m_per_deg
    lat = float(impact_lat_deg) + north / lat_scale_m_per_deg
    return lon, lat


def collect_map_coordinates(results):
    lons = []
    lats = []

    for result in results:
        model = result["model"]
        lons.extend(n.asarray(model["lon_deg"], dtype=float))
        lats.extend(n.asarray(model["lat_deg"], dtype=float))

        impact = result.get("impact")
        if impact is not None:
            traj = impact["trajectory"]
            lons.extend(n.asarray(traj["lon_deg"], dtype=float))
            lats.extend(n.asarray(traj["lat_deg"], dtype=float))
            lons.append(float(impact["impact_lon_deg"]))
            lats.append(float(impact["impact_lat_deg"]))

            impact_uncertainty = result.get("impact_uncertainty")
            if impact_uncertainty is not None:
                ellipse_lon, ellipse_lat = build_impact_uncertainty_ellipse_lon_lat(
                    impact["impact_lat_deg"],
                    impact["impact_lon_deg"],
                    impact_uncertainty["impact_horizontal_major_axis_1sigma_m"],
                    impact_uncertainty["impact_horizontal_minor_axis_1sigma_m"],
                    impact_uncertainty["impact_horizontal_major_axis_azimuth_deg"],
                )
                lons.extend(n.asarray(ellipse_lon, dtype=float))
                lats.extend(n.asarray(ellipse_lat, dtype=float))

    for _, lat_frag, lon_frag in RECOVERED_FRAGMENTS:
        lons.append(float(lon_frag))
        lats.append(float(lat_frag))

    return n.asarray(lons, dtype=float), n.asarray(lats, dtype=float)


def compute_map_extent(results):
    lons, lats = collect_map_coordinates(results)

    lon_min = float(n.nanmin(lons))
    lon_max = float(n.nanmax(lons))
    lat_min = float(n.nanmin(lats))
    lat_max = float(n.nanmax(lats))

    lon_span = max(lon_max - lon_min, 1.0)
    lat_span = max(lat_max - lat_min, 0.5)

    lon_pad = max(2.5, 0.12 * lon_span)
    lat_pad = max(1.5, 0.6 * lat_span)

    return [
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    ]


def format_fragment_id(label):
    prefix = "ballistic_fit_"
    token = label[len(prefix):] if label.startswith(prefix) else label
    return f"Fragment {token}"


def add_recovered_fragment_positions(ax):
    for i, (frag_id, _, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        ax.axvline(
            lon_frag,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=2,
            label="Recovered fragment longitude" if i == 0 else None,
        )

    ax.set_ylim(bottom=0.0)
    _, ymax = ax.get_ylim()
    label_y = 0.02 * ymax if ymax > 0.0 else 0.0

    for frag_id, _, lon_frag in RECOVERED_FRAGMENTS:
        ax.plot(
            lon_frag,
            0.0,
            marker="*",
            linestyle="None",
            color="gold",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            zorder=6,
        )
        ax.text(
            lon_frag + 0.01,
            label_y,
            frag_id,
            fontsize=7,
            rotation=90,
            ha="left",
            va="bottom",
            zorder=7,
        )


def make_dynamic_pressure(segment):
    return 0.5 * segment["rho_a_kg_m3"] * segment["relative_speed_m_s"] ** 2


def gather_valid_values(results, value_fn, positive_only=False):
    valid_values = []
    for result in results:
        segments = [result["model"]]
        impact = result.get("impact")
        if impact is not None:
            segments.append(impact["trajectory"])
        for segment in segments:
            values = n.asarray(value_fn(segment), dtype=float)
            mask = n.isfinite(values)
            if positive_only:
                mask &= values > 0.0
            if n.any(mask):
                valid_values.append(values[mask])
    return valid_values


def scatter_segment(ax, segment, value_fn, label, marker, cmap, norm, positive_only):
    values = n.asarray(value_fn(segment), dtype=float)
    mask = n.isfinite(values)
    if positive_only:
        mask &= values > 0.0
    if not n.any(mask):
        return None

    return ax.scatter(
        segment["lon_deg"][mask],
        segment["hgt_m"][mask] / 1e3,
        c=values[mask],
        s=22,
        cmap=cmap,
        norm=norm,
        marker=marker,
        linewidths=0.0,
        edgecolors="none",
        alpha=1.0,
        label=label,
        rasterized=True,
        zorder=4,
    )


def annotate_fragment_start_positions(ax, results):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = 0.015 * (x1 - x0)
    dy = 0.025 * (y1 - y0)

    for idx, result in enumerate(results):
        model = result["model"]
        start_lon = float(model["lon_deg"][0])
        start_hgt_km = float(model["hgt_m"][0] / 1e3)
        y_offset = dy if idx % 2 == 0 else -dy
        va = "bottom" if idx % 2 == 0 else "top"
        ax.text(
            start_lon + dx,
            start_hgt_km + y_offset,
            format_fragment_id(result["label"]),
            fontsize=8.5,
            fontweight="semibold",
            color=FIT_COLORS[idx % len(FIT_COLORS)],
            ha="left",
            va=va,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=9,
        )


def add_publication_lon_height_legend(ax):
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="0.35",
            markeredgewidth=0.0,
            color="0.35",
            label="Trajectory samples",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=6,
            markerfacecolor="0.35",
            markeredgewidth=0.0,
            color="0.35",
            label="Extrapolated samples",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            color="tab:blue",
            label="Estimated ground fall",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=9,
            markerfacecolor="gold",
            markeredgecolor="black",
            color="gold",
            label="Recovered fragment",
        ),
        Line2D(
            [],
            [],
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="Recovered fragment longitude",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )


def plot_lon_height_colored(
    results,
    value_fn,
    colorbar_label,
    title,
    cmap="viridis",
    log_color=False,
    positive_only=False,
    annotate_start_labels=False,
    publication_ready=False,
    save_path=None,
):
    if publication_ready:
        fig, ax = plt.subplots(figsize=(7.2, 4.9), dpi=200)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    valid_values = gather_valid_values(
        results,
        value_fn=value_fn,
        positive_only=positive_only,
    )

    norm = None
    if valid_values:
        values = n.concatenate(valid_values)
        vmin = float(n.nanmin(values))
        vmax = float(n.nanmax(values))
        if log_color and vmin > 0.0 and vmax > vmin:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        elif vmax > vmin:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    scatter_handle = None
    for idx, result in enumerate(results):
        model_marker, impact_marker = FIT_MARKERS[idx % len(FIT_MARKERS)]
        color = FIT_COLORS[idx % len(FIT_COLORS)]
        model_label = f"{result['label']} fit"
        impact_label = f"{result['label']} extrapolated"

        sc_model = scatter_segment(
            ax,
            result["model"],
            value_fn=value_fn,
            label=model_label,
            marker=model_marker,
            cmap=cmap,
            norm=norm,
            positive_only=positive_only,
        )
        if scatter_handle is None and sc_model is not None:
            scatter_handle = sc_model

        impact = result.get("impact")
        if impact is not None:
            sc_impact = scatter_segment(
                ax,
                impact["trajectory"],
                value_fn=value_fn,
                label=impact_label,
                marker=impact_marker,
                cmap=cmap,
                norm=norm,
                positive_only=positive_only,
            )
            if scatter_handle is None and sc_impact is not None:
                scatter_handle = sc_impact

            ax.plot(
                impact["impact_lon_deg"],
                impact["impact_hgt_m"] / 1e3,
                marker="*",
                linestyle="None",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=12,
                label=f"{result['label']} impact",
                zorder=7,
            )

    add_recovered_fragment_positions(ax)
    if annotate_start_labels:
        annotate_fragment_start_positions(ax, results)

    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, ax=ax)
        cbar.set_label(colorbar_label)
        if publication_ready:
            cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Height (km)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
    ax.tick_params(labelsize=8 if publication_ready else None)
    if publication_ready:
        add_publication_lon_height_legend(ax)
    else:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)


def plot_lon_lat(results):
    data_crs = ccrs.PlateCarree()
    extent = compute_map_extent(results)
    center_lon = 0.5 * (extent[0] + extent[1])
    center_lat = 0.5 * (extent[2] + extent[3])
    map_crs = ccrs.LambertConformal(
        central_longitude=center_lon,
        central_latitude=center_lat,
        standard_parallels=(center_lat - 2.0, center_lat + 2.0),
    )
    fig = plt.figure(figsize=(10, 7.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=map_crs)
    ax.set_extent(extent, crs=data_crs)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f2e8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dce9f5", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dce9f5", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="0.35", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.45", zorder=1)
    gl = ax.gridlines(
        crs=data_crs,
        draw_labels=False,
        linewidth=0.45,
        color="0.55",
        alpha=0.35,
        linestyle="--",
    )
    gl.xpadding = 6
    gl.ypadding = 6
    ax.spines["geo"].set_linewidth(0.8)
    ax.spines["geo"].set_edgecolor("0.35")
    uncertainty_label_added = False

    for idx, result in enumerate(results):
        color = FIT_COLORS[idx % len(FIT_COLORS)]
        label = result["label"]
        model = result["model"]
        ax.plot(
            model["lon_deg"],
            model["lat_deg"],
            "-",
            linewidth=2.0,
            color=color,
            label=f"{label} fit",
            zorder=4,
            transform=data_crs,
        )

        impact = result.get("impact")
        if impact is not None:
            ax.plot(
                impact["trajectory"]["lon_deg"],
                impact["trajectory"]["lat_deg"],
                "--",
                linewidth=1.8,
                color=color,
                label=f"{label} extrapolated",
                zorder=4,
                transform=data_crs,
            )
            ax.plot(
                impact["impact_lon_deg"],
                impact["impact_lat_deg"],
                marker="*",
                linestyle="None",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=12,
                label=f"{label} impact",
                zorder=7,
                transform=data_crs,
            )

            impact_uncertainty = result.get("impact_uncertainty")
            if impact_uncertainty is not None:
                ellipse_lon, ellipse_lat = build_impact_uncertainty_ellipse_lon_lat(
                    impact["impact_lat_deg"],
                    impact["impact_lon_deg"],
                    impact_uncertainty["impact_horizontal_major_axis_1sigma_m"],
                    impact_uncertainty["impact_horizontal_minor_axis_1sigma_m"],
                    impact_uncertainty["impact_horizontal_major_axis_azimuth_deg"],
                )
                ax.plot(
                    ellipse_lon,
                    ellipse_lat,
                    color="0.45",
                    linewidth=1.3,
                    alpha=0.95,
                    zorder=5,
                    label="Impact uncertainty (1$\\sigma$)" if not uncertainty_label_added else None,
                    transform=data_crs,
                )
                uncertainty_label_added = True

    for i, (frag_id, lat_frag, lon_frag) in enumerate(RECOVERED_FRAGMENTS):
        ax.plot(
            lon_frag,
            lat_frag,
            marker="*",
            linestyle="None",
            color="gold",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=9,
            label="Recovered fragment" if i == 0 else None,
            zorder=6,
            transform=data_crs,
        )
        ax.text(
            lon_frag + 0.02,
            lat_frag + 0.02,
            frag_id,
            fontsize=7.5,
            ha="left",
            va="bottom",
            zorder=7,
            transform=data_crs,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

    ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
        columnspacing=1.0,
        handlelength=2.8,
    )
    fig.tight_layout()


def main():
    results = [load_fit_result(path) for path in FIT_FILES]
    energy_loss_pdf = Path(__file__).with_name("specific_energy_loss_rate_publication.pdf")

    plot_lon_lat(results)

    plot_lon_height_colored(
        results,
        value_fn=make_dynamic_pressure,
        colorbar_label="Dynamic pressure (Pa)",
        title="Dynamic pressure",
        cmap="plasma",
        log_color=True,
        positive_only=True,
    )

    plot_lon_height_colored(
        results,
        value_fn=lambda segment: segment["relative_speed_m_s"],
        colorbar_label=r"Velocity relative to atmosphere (m s$^{-1}$)",
        title="Velocity relative to atmosphere",
        cmap="viridis",
        log_color=False,
        positive_only=False,
    )

    plot_lon_height_colored(
        results,
        value_fn=lambda segment: segment["specific_energy_loss_rate_w_kg"],
        colorbar_label=r"Energy loss rate per unit mass (W kg$^{-1}$)",
        title=None,
        cmap="inferno",
        log_color=True,
        positive_only=True,
        annotate_start_labels=True,
        publication_ready=True,
        save_path=energy_loss_pdf,
    )

    plt.show()


if __name__ == "__main__":
    main()

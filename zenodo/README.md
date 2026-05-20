# Replication Data for "Optical and VHF Radar Observations of the February 2025 Falcon 9 Upper-Stage Re-entry"

Authors:
Juha Vierinen, Dabrowka Knach, Jorge L. Chau, Gerd Baumgarten, Devin Huyghebaert, Matthias Clahsen, Nico Pfeffer, Toralf Renkwitz, Robin Wing, Kenneth S. Obenberger, and Bjorn Gustavsson

This directory is the staged Zenodo release for the manuscript *Optical and VHF Radar Observations of the February 2025 Falcon 9 Upper-Stage Re-entry*. It contains both primary data products and smaller derivative files used to reproduce figures in the paper.

## Directory Guide

### `range_doppler_mf/`

Matched-filter radar output products for the Juliusruh and Kuehlungsborn processing chains:

- `range_doppler_mf_jr.h5`
- `range_doppler_mf_kb.h5`

These files contain gridded range-Doppler outputs. A representative file includes:

- `P`: matched-filter power grid
- `N`: noise estimate grid
- `D`: Doppler grid or Doppler-related matched-filter output
- `t0`: reference start time

The array shape is roughly `11999 x 999`, so these are dense radar products rather than event lists.

### `radar_geodetic_data/`

Geolocated radar detections for the eight SIMONe transmitter-receiver links:

- `geodetic_data_jruh_bornholm_20250219_corr.h5`
- `geodetic_data_jruh_bornim_20250219_corr.h5`
- `geodetic_data_jruh_hagenow_20250219_corr.h5`
- `geodetic_data_jruh_moitin_20250219_corr.h5`
- `geodetic_data_kborn_bornholm_20250219_corr.h5`
- `geodetic_data_kborn_bornim_20250219_corr.h5`
- `geodetic_data_kborn_hagenow_20250219_corr.h5`
- `geodetic_data_kborn_moitin_20250219_corr.h5`

Each file contains event-level detections with fields such as:

- `time_unix`
- `range_m`
- `latitude`
- `longitude`
- `altitude_m`
- `doppler_hz`
- `peak_power_db`
- `peak_idx`
- `bragg_enu`

These are compact geophysical detection tables and are much easier to work with than the full decoded SIMONe data.

### `fragments/triangulated_positions/`

Triangulated optical fragment position files derived from paired camera observations. Each file corresponds to one triangulated point in time for one fragment-track pairing.

A representative file contains:

- `time`
- `pos_est`: estimated ECEF position vector
- `pos_err`: scalar position uncertainty
- `geoloc`: geodetic location

The filenames encode the fragment id, camera pairing, and timestamp.

### `ballistic_fit_sharedstart/`

Shared-start ballistic fit solutions used in the manuscript analysis. These are the main compact trajectory-model products for the optical fragments.

Included files:

- `ballistic_fit_sharedstart_1.h5`
- `ballistic_fit_sharedstart_2.h5`
- `ballistic_fit_sharedstart_5_1.h5`
- `ballistic_fit_sharedstart_9_1.h5`
- `ballistic_fit_sharedstart_k_a_7_2.h5`
- `ballistic_fit_sharedstart_t_i_7_2.h5`
- `ballistic_fit_sharedstart_z_w_p_1.h5`

These files contain both measurements and model outputs. Important contents include:

- top-level observed trajectories:
  - `times_unix`
  - `pos_ecef`
  - `pos_eci`
  - `pos_ecef_err`
- fit configuration:
  - `fit_ids`
  - `shared_start_id`
  - `fit_parameter_names`
- atmospheric model data:
  - `density_profile_altitude_m`
  - `density_profile_rho_kg_m3`
  - reference latitude, longitude, and time
- forward model outputs under `model/`:
  - `times_model`
  - `lat_deg`
  - `lon_deg`
  - `hgt_m`
  - `pos_eci`
  - `vel_eci`
  - `speed_m_s`
  - `relative_speed_m_s`
  - `specific_energy_loss_rate_w_kg`
  - `rho_a_kg_m3`
  - `B_model`
- extrapolated impact information under `impact/`:
  - `impact_time_unix`
  - `impact_lat_deg`
  - `impact_lon_deg`
  - `impact_hgt_m`
  - `impact_speed_m_s`
  - `impact_relative_speed_m_s`
  - `impact_specific_energy_loss_rate_w_kg`
  - `impact_pos_eci`
  - `impact_vel_eci`
  - `impact/trajectory/` with the extrapolated trajectory history
- uncertainty information under `impact_uncertainty/`
- velocity-fit diagnostics under `velocity_constraints/`
- optimizer summaries under `optimizer/`, `optimizer_era5/`, and `optimizer_zero_wind/`

These are among the most important files in the bundle for trajectory interpretation and figure reproduction.

### `simone/decoded_files/`

Full decoded SIMONe radar data products. This is the largest part of the Zenodo bundle, around `14 GB`.

These files are the underlying decoded radar HDF5 products for links such as:

- `mmaria_decoded_jruh_*_3255_20250219_*.h5`
- `mmaria_decoded_kborn_*_3255_20250219_*.h5`

These are the high-volume source files behind later derived radar plots, geodetic detections, and imaging summaries.

### `simone/figures_miso/`

Compact radar-imaging summary products from the Koki workflow. This directory includes:

- `summary_data_kborn_hagenow_20250219_miso_ref_tx_corr.h5`
- `summary_geolocation_combined_kborn_hagenow_20250219_miso_ref_tx_corr.png`
- `summary_range_time_fit_kborn_hagenow_20250219_miso_ref_tx_corr.png`
- `summary_time_power_doppler_plot_kborn_hagenow_20250219_miso_ref_tx_corr.png`

The HDF5 summary file is the compact numeric source used by some of the radar-imaging plots.

### `simone/output_hdf5/`

Geodetic radar-imaging output from the Koki workflow. The included file

- `geodetic_data_kborn_hagenow_20250219_miso_ref_tx_corr.h5`

contains geolocated detections similar in spirit to the `radar_geodetic_data/` products, with fields such as:

- `time_unix`
- `range_m`
- `latitude`
- `longitude`
- `altitude_m`
- `doppler_hz`
- `peak_power_db`
- `peak_idx`

### `figure_data/`

Compact sidecar files for reproducing the main data-driven manuscript figures. These are much smaller than the raw analysis products and are intended to provide the minimal numeric inputs needed for plotting.

Included files:

- `fig_map_falcon9.h5`
- `fragment_visibility.h5`
- `fragment_radar_height_hist.h5`
- `optical_radar_overview.h5`
- `snr_all_links_summary.h5`
- `rcs_doppler_triptych_sidecar.h5`

Representative contents:

- `fig_map_falcon9.h5`
  - `camera_sites/`
  - `optical/fragments/`
  - `radar_links/`
  - `simone_stations/tx` and `simone_stations/rx`
  - `recovered_fragments/`
  - `predicted_impacts/`
- `fragment_visibility.h5`
  - fragment times, positions, and altitudes used for the visibility plot
- `fragment_radar_height_hist.h5`
  - optical initial heights
  - radar detection heights
  - histogram bin edges
- `optical_radar_overview.h5`
  - optical fragments
  - radar detections
  - compact copies of the shared-start ballistic model outputs used in overlay panels
- `snr_all_links_summary.h5`
  - decimated `S/N + 1` grids for all eight radar links
- `rcs_doppler_triptych_sidecar.h5`
  - compact sidecar written by the triptych plot script for the RCS and Doppler figure

There is also a helper script:

- `export_figure_data.py`

which was used to generate several of these compact sidecars.

### `allsky7/`

AllSky7 optical camera material staged for the Zenodo bundle.

This subtree currently contains:

- `camera_obs/`
  - Original camera observation videos grouped by station, for example `AMS16/`, `AMS21/`, `AMS22/`, `AMS52/`, `AMS76/`, `AMS88/`, `AMS95/`, and others.
  - The files are mostly `*.mp4` observations from the 19 February 2025 event.

- `saved_calib/`
  - MATLAB calibration files for the subset of cameras used in calibration and validation, for example:
    - `AMS16_5.mat`
    - `AMS16_6.mat`
    - `AMS213_5.mat`
    - `AMS216_5.mat`
    - `AMS22_1.mat`
    - `AMS22_5.mat`
    - `AMS238_6.mat`
    - `AMS35_7.mat`
    - `AMS52_5.mat`
    - `AMS76_1.mat`
    - `AMS76_2.mat`
    - `AMS88_1.mat`
    - `AMS88_2.mat`
    - `AMS95_5.mat`
    - `ams21_1.mat`
    - `ams21_5.mat`
    - `ams22_2.mat`
    - `ams62_5.mat`
  - These `.mat` files contain calibration products such as:
    - `optpar`
    - `long_lat`
    - `az`
    - `ze`
    - sometimes `obs`, `epix`, or `t_obs`
  - The directory also currently includes two image assets retained for calibration illustration:
    - `star_calib_ex.png`
    - `scattererror238_6_last.jpg`

### `figures/`

Image-based manuscript figure assets copied from `falcon9_paper/figures`. These are not numeric data tables; they are raster figures used directly in the manuscript.

Currently included:

- `fragment_times_good.png`
- `star_calib_ex.png`
- `scattererror238_6_last.jpg`

## How To Use This Bundle

- Use `simone/decoded_files/` if you need the full decoded radar source data.
- Use `radar_geodetic_data/` or `simone/output_hdf5/` if you want compact geolocated radar detections.
- Use `fragments/triangulated_positions/` for camera-derived fragment positions.
- Use `ballistic_fit_sharedstart/` for modeled fragment trajectories, velocities, energy-loss estimates, and impact extrapolations.
- Use `figure_data/` if your goal is to reproduce manuscript figures with minimal extra processing.
- Use `allsky7/camera_obs/` for the original optical videos.
- Use `allsky7/saved_calib/` for calibration `.mat` files and the retained calibration illustration images.
- Use `figures/` for manuscript image assets already prepared for visual inspection.

## Notes

- The bundle mixes primary analysis products with smaller derivative products intended for plotting and publication support.
- The largest component is `simone/decoded_files/`.
- Some subdirectories contain their own `README.md` files with more specific descriptions.

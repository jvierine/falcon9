import h5py
import numpy as np
import numpy as n
import os
import scipy.constants as sc
import scipy as sp
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.ndimage import uniform_filter1d, median_filter
import glob
import argparse
# import sys  # Unused
from mpl_toolkits.axes_grid1 import make_axes_locatable
from beamforming import (beamform_coherences, detect_one_target_bartlet_clean_nlls, 
                         estimate_direction_cosines_nlls, detect_two_targets_capon_clean_nlls,
                         detect_two_targets_capon_clean_batch, estimate_direction_cosines_nlls_two_targets)
from coordinates import geolocation_from_bistatic_peak, wgs84_lla_to_ecef
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from read_hdf5 import read_hdf5_data



""" Examples of calls
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_moitin" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_moitin" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0 
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_bornholm" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_bornholm" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_bornim" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_bornim" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_miso_figures/' --detect_one_nlls True --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold 0 --read_summary 0

python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_5ms/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_5ms_figures/' --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures/' --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold -10 --num_peaks 2 --detect_two_nlls True --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_bornim" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures/' --v_doppler_min -10 --v_doppler_max 10 --snr_summary_threshold -10 --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures/' --v_doppler_min -20 --v_doppler_max 20 --snr_summary_threshold -0  --detect_one_nlls True --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures_5_smooth/' --v_doppler_min -20 --v_doppler_max 20 --snr_summary_threshold -20  --detect_one_nlls True --read_summary 0
python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures/' --v_doppler_min -20 --v_doppler_max 20  --detect_one_nlls True --range_min 200 --range_max 400 --time_min 03:45:45 --time_max 03:46:45 --snr_summary_threshold -0 --read_summary 1

For Falcon9 paper

python correlate_tx_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_miso_figures/' --v_doppler_min -20 --v_doppler_max 20  --detect_one_nlls True --range_min 200 --range_max 400 --time_min 03:45:40 --time_max 03:46:35 --snr_summary_threshold 1 --read_summary 1
"""

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


if __name__ == "__main__":
    
    # Default values
    default_year = 2026
    default_month = 1
    default_day = 19
    default_read_summary = False
    default_dpath = '/Volumes/KCH_4TB_IAP/SIMONe/Eregion/'
    default_gpath = '/Volumes/KCH_4TB_IAP/SIMONe/Eregion_figures/'
    default_link_name = 'Jic_Azp'
    default_event = 'EEJ'
    default_read_residuals = False  #True
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Process SIMONe decoded data HDF5 files.')
    parser.add_argument('--year', type=int, default=default_year, help=f'Year to process (default: {default_year})')
    parser.add_argument('--month', type=int, default=default_month, help=f'Month to process (default: {default_month})')
    parser.add_argument('--day', type=int, default=default_day, help=f'Day to process (default: {default_day})')
    parser.add_argument('--read_summary', type=str2bool, default=default_read_summary, help='Read from summary file if exists (True/False)')
    parser.add_argument('--dpath', type=str, default=default_dpath, help=f'Input data path (default: {default_dpath})')
    parser.add_argument('--gpath', type=str, default=default_gpath, help=f'Output figures path (default: {default_gpath})')
    parser.add_argument('--link_name', type=str, default=default_link_name, help='Radar link name to use in filenames and plots')
    parser.add_argument('--event', type=str, default=default_event, help='Event name to select predefined parameters (e.g., EEJ)')
    parser.add_argument('--v_doppler_min', type=float, default=-200, help='Minimum Doppler velocity for plots (Hz)')
    parser.add_argument('--v_doppler_max', type=float, default=200, help='Maximum Doppler velocity for plots (Hz)')
    parser.add_argument('--snr_summary_threshold', type=float, default=None, help='Additional SNR threshold (dB) when reading from summary')
    parser.add_argument('--read_residuals', type=str2bool, default=default_read_residuals, help='Read residuals from HDF5 if they exist (True/False)')
    parser.add_argument('--detect_one_nlls', type=str2bool, default=False, help='Use NLLS refinement for single peak detection (True/False)')
    parser.add_argument('--num_peaks', type=int, default=1, help='Number of peaks to detect (1 or 2)')
    parser.add_argument('--detect_two_nlls', type=str2bool, default=False, help='Use Capon-CLEAN-NLLS for two-peak detection (True/False)')
    parser.add_argument('--range_min', type=float, default=None, help='Minimum scattering range to process (km)')
    parser.add_argument('--range_max', type=float, default=None, help='Maximum scattering range to process (km)')
    parser.add_argument('--time_min', type=str, default=None, help='Start time for summary plots (HH:MM)')
    parser.add_argument('--time_max', type=str, default=None, help='End time for summary plots (HH:MM)')
    parser.add_argument('--plot_peak_rcs', type=str2bool, default=True, help='Plot RCS from peak SNR (beamforming peaks) in the summary plot (True/False)')
    
    args = parser.parse_args()
    
    event = args.event
    link_name = args.link_name
    link_prefix = f"{link_name}_" if link_name else ""
    
    dpath = args.dpath
    delta_range = 3.00
    
    # Baseline/Default values
    num_parts = 3           # How many parts of the data to process
    smooth_val_time = 100
    snr_time_vmin = 0
    snr_time_vmax = 10
    geo_power_vmin = 40
    geo_power_vmax = 60
    geo_alt_min = 90  # km
    geo_alt_max = 110 # km
    range_min = 180
    range_max = 400         # 500
    geo_delta_deg = 2.0
    grid_size_bf = 256  #64
    clean_gain = 0.9

    # Event-specific overrides
    if event:
        if event.upper() == 'EEJ':
            print(f"Applying parameters for event: {event}")
            num_parts = 3           # How many parts of the data to process
            smooth_val_time = 10
            snr_time_vmin = 0
            snr_time_vmax = 10
            geo_power_vmin = 40
            geo_power_vmax = 60
            geo_alt_min = 90  # km
            geo_alt_max = 110 # km
            range_min = 180
            range_max = 400         # 500
            geo_delta_deg = 2.0
        elif event.upper() == 'FALCON':
            print(f"Applying parameters for event: {event}")
            num_parts = 3           # How many parts of the data to process
            smooth_val_time = 50# 100
            snr_time_vmin = -3
            snr_time_vmax = 20
            geo_power_vmin = 40
            geo_power_vmax = 60
            geo_alt_min = 40  # km
            geo_alt_max = 80 # km
            range_min = 100
            range_max = 600 
            geo_delta_deg = 2.5
            grid_size_bf = 256 #512  #64

        else:
            print(f"Warning: Unknown event '{event}'. Using default parameters.")
    
    # User-defined range overrides
    if args.range_min is not None:
        range_min = args.range_min
        print(f"Applying user-defined range_min: {range_min}")
    if args.range_max is not None:
        range_max = args.range_max
        print(f"Applying user-defined range_max: {range_max}")
        
    if args.time_min is not None:
        print(f"Applying user-defined time_min: {args.time_min}")
    if args.time_max is not None:
        print(f"Applying user-defined time_max: {args.time_max}")
        
    cmap_time = 'viridis'
    cmap_coherence = 'terrain'
    snr_threshold = 1 #0.5 #1  # dB
    ch_a = 0
    ch_b = 5
    # Use a moderate grid size in beamforming for speed
    
    year = args.year
    month = args.month
    day = args.day
    date_str = f"{year:04d}{month:02d}{day:02d}"
    obs_date_str_actual = date_str
    
    gpath = args.gpath
    read_summary = args.read_summary
    miso_suffix = "miso_ref_tx"
    summary_file_to_read = f'summary_data_{link_prefix}{date_str}_{miso_suffix}_corr.h5' # Specific file to read from gpath
    
    # Ensure the figures directory exists 
    if not os.path.exists(gpath):
        os.makedirs(gpath)
        print(f"Created directory: {gpath}")
    
    # Ensure the geodetic hdf5 directory exists under the current working directory
    geodetic_path = "output_hdf5"
    if not os.path.exists(geodetic_path):
        os.makedirs(geodetic_path)
        print(f"Created directory: {geodetic_path}")

    # Find h5 files in dpath for the given date
    file_list = sorted(glob.glob(os.path.join(dpath, f"*{link_prefix}*{date_str}*.h5")))
    print(f"Found {len(file_list)} files to process for {date_str}.")

    all_summed_power = []
    all_doppler = []
    all_taxis = []
    all_taxis_spectra = []
    
    all_peak_L = []
    all_peak_M = []
    all_peak_P = []
    
    all_peak_L2 = []
    all_peak_M2 = []
    all_peak_P2 = []
    
    global_tx_name = ""
    global_rx_name = ""
    wavelength = None

    # Summary Plot and Data Loading
    summary_power = None
    summary_doppler = None
    summary_peak_L = None
    summary_peak_M = None
    summary_peak_P = None
    
    summary_peak_L2 = None
    summary_peak_M2 = None
    summary_peak_P2 = None
    
    data_loaded_from_summary = False

    global_tx_ant_coords = None
    global_tx_gps = None
    global_rx_gps = None
    num_tx = 6 # Default for Kborn

    if read_summary:
        summary_h5_path = os.path.join(gpath, summary_file_to_read)
        if os.path.exists(summary_h5_path):
            print(f"\nReading summary from {summary_h5_path}...")
            try:
                with h5py.File(summary_h5_path, 'r') as sf:
                    summary_power = sf['summary_power'][:]
                    taxis_timestamps = sf['taxis'][:]
                    all_taxis = [datetime.fromtimestamp(t, timezone.utc) for t in taxis_timestamps]
                    if 'summary_doppler' in sf:
                        summary_doppler = sf['summary_doppler'][:]
                    
                    if 'summary_peak_L' in sf:
                        summary_peak_L = sf['summary_peak_L'][:]
                    if 'summary_peak_M' in sf:
                        summary_peak_M = sf['summary_peak_M'][:]
                    if 'summary_peak_P' in sf:
                        summary_peak_P = sf['summary_peak_P'][:]
                    
                    if 'vel_axis' in sf:
                        global_vel_axis = sf['vel_axis'][:]

                    ranges = sf['ranges'][:]
                    global_tx_name = sf.attrs.get('tx_name', "Unknown TX")
                    global_rx_name = sf.attrs.get('rx_name', "Unknown RX")
                    
                    if 'tx_ant_coords' in sf.attrs:
                        global_tx_ant_coords = sf.attrs['tx_ant_coords']
                        num_tx = len(global_tx_ant_coords)
                        print(f"Detected {num_tx} transmitters from summary metadata.")
                    if 'tx_gps' in sf.attrs:
                        global_tx_gps = sf.attrs['tx_gps']
                    if 'rx_gps' in sf.attrs:
                        global_rx_gps = sf.attrs['rx_gps']
                    
                    if 'wavelength' in sf.attrs:
                        wavelength = sf.attrs['wavelength']
                        print(f"Loaded wavelength from summary: {wavelength} m")
                        
                print("Summary data loaded successfully.")
                data_loaded_from_summary = True
            except Exception as e:
                print(f"Error reading summary file: {e}")
                summary_power = None
                print("Falling back to raw file processing.")
        else:
            print(f"\nSummary file {summary_h5_path} not found.")
            print("Falling back to raw file processing.")


    if not data_loaded_from_summary:
        for file_path in file_list: # Process all files for the given date
            file_name_orig = os.path.basename(file_path)
            print(f"\nProcessing {file_name_orig}...")
            
            full_voltage, full_start_time, full_duration, wavelength, tx_name, rx_name, tx_ant_coords, tx_gps, rx_gps, chunk_noise_power, range_offset = read_hdf5_data(file_path, read_residuals=args.read_residuals, return_rx_coords=False)
            
            if full_voltage is None:
                continue
                
            global_tx_name = tx_name
            global_rx_name = rx_name
            global_tx_ant_coords = tx_ant_coords
            global_tx_gps = tx_gps
            global_rx_gps = rx_gps
            
            full_num_samples = full_voltage.shape[-1]
            part_n = full_num_samples // num_parts
            
            # Ranges (remains the same for all parts)
            num_ranges = full_voltage.shape[2]
            ranges = np.arange(num_ranges) * delta_range + range_offset
            
            # Process each part
            for h in range(num_parts):
                idx_start = h * part_n
                idx_end = (h + 1) * part_n if h < num_parts - 1 else full_num_samples
                
                voltage = full_voltage[..., idx_start:idx_end]
                num_samples = voltage.shape[-1]
                
                # Time metadata for this part
                # start_time is in nanoseconds
                part_duration = full_duration / float(num_parts)
                start_time = full_start_time + h * part_duration
                
                # Create taxis
                taxis_ns = np.linspace(start_time, start_time + part_duration, num_samples)
                taxis = [datetime.fromtimestamp(t / 1e9, timezone.utc) for t in taxis_ns]
                
                # Number of time bins within this part
                n_samples_bin = smooth_val_time
                n_bins = num_samples // n_samples_bin
                num_samples_adj = n_bins * n_samples_bin # Trim to integer number of bins
                
                # Reshape to (nrxs, num_tx, num_ranges, n_bins, n_samples_bin)
                # voltage is (nrxs, num_tx, num_ranges, num_samples)
                nrxs, num_tx, num_ranges, _ = voltage.shape
                v_bin = voltage[..., :num_samples_adj].reshape(nrxs, num_tx, num_ranges, n_bins, n_samples_bin)
                
                # 1. Total Power per bin (averaged over Rx, Tx, and samples within bin)
                # Power shape: (num_ranges, n_bins)
                summed_power = np.mean(np.abs(v_bin)**2, axis=(0, 1, 4))
                
                # 1b. Doppler estimation (Pulse-to-pulse)
                # Calculate lag-1 autocorrelation averaged over Rx and Tx
                # Sample axis is 4. Lag-1: v(n) * conj(v(n-1))
                prt = (taxis_ns[1] - taxis_ns[0]) / 1e9 # PRT in seconds
                r1_bin = np.mean(v_bin[..., 1:] * np.conj(v_bin[..., :-1]), axis=(0, 1, 4))
                # Doppler Frequency: angle(R1) / (2*pi*PRT)
                summed_doppler = np.angle(r1_bin) / (2 * np.pi * prt)
                
                # Mask Doppler estimation based on SNR later (after noise estimation)
                
                # 2. Time-domain Cross-Correlation Matrix (ntxs x ntxs) per range and bin
                # sum over Rx (axis 0) and samples (axis 4)
                summed_cross_corr = np.zeros((num_tx, num_tx, num_ranges, n_bins), dtype=complex)
                for i in range(num_tx):
                    for j in range(num_tx):
                        # Mean over receivers and samples within bin
                        summed_cross_corr[i, j] = np.mean(v_bin[:, i] * np.conj(v_bin[:, j]), axis=(0, 3))

                # 3. Noise estimation
                # Since we don't have spectra, we use the time-domain power
                noise_time = np.median(summed_power)
                noise_time = max(noise_time, 1e-12)
                
                # Save for summary (decimate to one point per bin for the intensity plot)
                all_summed_power.append(summed_power)
                all_doppler.append(summed_doppler)
                # taxis_bins
                taxis_ns_bins = taxis_ns[:num_samples_adj:n_samples_bin]
                taxis_bins = [datetime.fromtimestamp(t / 1e9, timezone.utc) for t in taxis_ns_bins]
                all_taxis.extend(taxis_bins)

                # -------------------------------------------------------------------------
                # Beamforming Peak Estimation (Zero-Lag)
                # -------------------------------------------------------------------------
                snr_bins = 10 * np.log10(np.maximum(summed_power / noise_time, 1e-12))
                
                # Mask summed_doppler by SNR threshold
                summed_doppler[snr_bins <= snr_threshold] = np.nan
                
                peak_L = np.full((num_ranges, n_bins), np.nan)
                peak_M = np.full((num_ranges, n_bins), np.nan)
                peak_P = np.full((num_ranges, n_bins), np.nan)
                
                peak_L2 = np.full((num_ranges, n_bins), np.nan)
                peak_M2 = np.full((num_ranges, n_bins), np.nan)
                peak_P2 = np.full((num_ranges, n_bins), np.nan)
                
                if wavelength is not None and global_tx_ant_coords is not None:
                    valid_mask = snr_bins > snr_threshold
                    
                    if np.any(valid_mask):
                        n_valid = np.sum(valid_mask)
                        print(f"  Running zero-lag beamforming on {n_valid} points...")
                        
                        # Baselines
                        bl_vectors = []
                        bl_indices = []
                        for i in range(num_tx):
                            for j in range(i + 1, num_tx):
                                bl = global_tx_ant_coords[i] - global_tx_ant_coords[j]
                                bl_vectors.append(bl)
                                bl_indices.append((i, j))
                                
                        if bl_vectors:
                            bl_vectors = np.array(bl_vectors)
                            batch_coherences = []
                            for (i, j) in bl_indices:
                                cs_slice = summed_cross_corr[i, j]
                                vals = cs_slice[valid_mask]
                                batch_coherences.append(vals)
                                
                            batch_coherences = np.array(batch_coherences) # (N_bl, N_valid)
                            
                            # Process in batches to save memory
                            bf_batch_size = 500
                            l_max_arr = np.zeros(n_valid)
                            m_max_arr = np.zeros(n_valid)
                            p_max_arr = np.zeros(n_valid)
                            
                            l_max2_arr = np.full(n_valid, np.nan)
                            m_max2_arr = np.full(n_valid, np.nan)
                            p_max2_arr = np.full(n_valid, np.nan)
                            
                            for b_start in range(0, n_valid, bf_batch_size):
                                b_end = min(b_start + bf_batch_size, n_valid)
                                current_batch = batch_coherences[:, b_start:b_end] # (N_bl, N_batch)
                                n_batch = b_end - b_start
                                
                                # Standard Bartlett peak finding first as initial guess
                                L_grid, M_grid, P_batch = beamform_coherences(current_batch, bl_vectors, wavelength, grid_size=grid_size_bf)
                                
                                P_flat = P_batch.reshape(n_batch, -1)
                                idx_max = np.nanargmax(P_flat, axis=1)
                                l_flat = L_grid.flatten()
                                m_flat = M_grid.flatten()

                                if args.num_peaks == 2 and args.detect_two_nlls:
                                    print("Batched Capon-CLEAN for two peak initial guesses...")
                                    (l1_guesses, m1_guesses), (l2_guesses, m2_guesses) = detect_two_targets_capon_clean_batch(
                                        current_batch, global_tx_ant_coords, wavelength, grid_size=grid_size_bf,clean_gain=clean_gain
                                    )
                                    
                                    print("Refining each sample in the batch individually for TWO peaks NLLS")
                                    for idx_in_batch in range(n_batch):
                                        sample_coh = current_batch[:, idx_in_batch]
                                        p1, p2, a1, a2, succ = estimate_direction_cosines_nlls_two_targets(
                                            sample_coh, bl_vectors, wavelength, 
                                            (l1_guesses[idx_in_batch], m1_guesses[idx_in_batch]),
                                            (l2_guesses[idx_in_batch], m2_guesses[idx_in_batch])
                                        )
                                        l_max_arr[b_start + idx_in_batch] = p1[0]
                                        m_max_arr[b_start + idx_in_batch] = p1[1]
                                        p_max_arr[b_start + idx_in_batch] = np.abs(a1)**2
                                        
                                        l_max2_arr[b_start + idx_in_batch] = p2[0]
                                        m_max2_arr[b_start + idx_in_batch] = p2[1]
                                        p_max2_arr[b_start + idx_in_batch] = np.abs(a2)**2
                                
                                elif args.detect_one_nlls:
                                    # Refine each sample in the batch individually for NLLS
                                    print("Refining each sample in the batch individually for NLLS")
                                    for idx_in_batch in range(n_batch):
                                        sample_coh = current_batch[:, idx_in_batch]
                                        l_init, m_init = l_flat[idx_max[idx_in_batch]], m_flat[idx_max[idx_in_batch]]
                                        
                                        l_est, m_est, a_est, succ_est = estimate_direction_cosines_nlls(
                                            sample_coh, bl_vectors, wavelength, initial_guess=(l_init, m_init)
                                        )
                                        l_max_arr[b_start + idx_in_batch] = l_est
                                        m_max_arr[b_start + idx_in_batch] = m_est
                                        p_max_arr[b_start + idx_in_batch] = np.abs(a_est)**2
                                else:
                                    l_max_arr[b_start:b_end] = l_flat[idx_max]
                                    m_max_arr[b_start:b_end] = m_flat[idx_max]
                                    p_max_arr[b_start:b_end] = P_flat[np.arange(n_batch), idx_max]
                                    
                                    if args.num_peaks == 2:
                                        # Simple Bartlett Masking fallback if detect_two_nlls is False
                                        for idx_in_batch in range(n_batch):
                                            # Distance squared from first peak
                                            dist_sq = (L_grid - l_flat[idx_max[idx_in_batch]])**2 + (M_grid - m_flat[idx_max[idx_in_batch]])**2
                                            P_masked = P_batch[idx_in_batch].copy()
                                            P_masked[dist_sq < 0.1**2] = np.nan
                                            
                                            idx_max2 = np.nanargmax(P_masked.flatten())
                                            l_max2_arr[b_start + idx_in_batch] = L_grid.flatten()[idx_max2]
                                            m_max2_arr[b_start + idx_in_batch] = M_grid.flatten()[idx_max2]
                                            p_max2_arr[b_start + idx_in_batch] = P_masked.flatten()[idx_max2]
                            
                            peak_L[valid_mask] = l_max_arr
                            peak_M[valid_mask] = m_max_arr
                            peak_P[valid_mask] = p_max_arr  + num_tx*summed_power[valid_mask]
                            
                            peak_L2[valid_mask] = l_max2_arr
                            peak_M2[valid_mask] = m_max2_arr
                            peak_P2[valid_mask] = p_max2_arr + num_tx*summed_power[valid_mask]
                            
                all_peak_L.append(peak_L)
                all_peak_M.append(peak_M)
                all_peak_P.append(peak_P)
                
                all_peak_L2.append(peak_L2)
                all_peak_M2.append(peak_M2)
                all_peak_P2.append(peak_P2)
                
                # Time axis for geolocation chunks (middle of the part)
                all_taxis_spectra.append(taxis[len(taxis)//2])

                # Plotting
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                ax1, ax2, ax3 = axs
                
                # Left: Time Power/Noise
                extent_bin = [mdates.date2num(taxis_bins[0]), mdates.date2num(taxis_bins[-1]), ranges[0], ranges[-1]]
                im1 = ax1.imshow(snr_bins, aspect='auto', origin='lower', cmap=cmap_time, extent=extent_bin, vmin=snr_time_vmin, vmax=snr_time_vmax, interpolation='nearest')
                plt.colorbar(im1, ax=ax1, label='SNR (dB)', pad=0.01)
                ax1.xaxis_date(tz=timezone.utc)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
                ax1.set_title('Time-Binned SNR (Zero-Lag)')
                ax1.set_ylabel('Total Range (km)')

                # Middle: Doppler Shift
                im2 = ax2.imshow(summed_doppler, aspect='auto', origin='lower', cmap='RdBu_r', extent=extent_bin, vmin=args.v_doppler_min, vmax=args.v_doppler_max, interpolation='nearest')
                plt.colorbar(im2, ax=ax2, label='Doppler Shift (Hz)', pad=0.01)
                ax2.xaxis_date(tz=timezone.utc)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
                ax2.set_title('Time-Binned Doppler Shift')
                ax2.set_ylabel('Total Range (km)')

                # Right: Peak L/M Scatter
                L_flat = peak_L.flatten()
                M_flat = peak_M.flatten()
                P_flat = peak_P.flatten()
                mask_valid = ~np.isnan(L_flat) & ~np.isnan(M_flat)
                
                if np.any(mask_valid):
                    # Show power or just distribution
                    sc = ax3.scatter(L_flat[mask_valid], M_flat[mask_valid], c=10*np.log10(np.maximum(P_flat[mask_valid], 1e-12)), 
                                   cmap='viridis', s=5, alpha=0.5, marker='.', label='Peak 1')
                    
                    if args.num_peaks == 2:
                        L2_flat = peak_L2.flatten()
                        M2_flat = peak_M2.flatten()
                        P2_flat = peak_P2.flatten()
                        mask2 = ~np.isnan(L2_flat) & ~np.isnan(M2_flat)
                        if np.any(mask2):
                             ax3.scatter(L2_flat[mask2], M2_flat[mask2], c='orange', s=5, alpha=0.5, marker='x', label='Peak 2')
                    
                    theta = np.linspace(0, 2*np.pi, 201)
                    ax3.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
                    plt.colorbar(sc, ax=ax3, label='Peak 1 Power (dB)')
                    ax3.set_xlim(-1, 1)
                    ax3.set_ylim(-1, 1)
                    ax3.set_aspect('equal')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend(loc='lower left', fontsize='small')
                else:
                    ax3.text(0.5, 0.5, 'No valid peaks', ha='center', va='center')
                    
                ax3.set_title(f'Zero-Lag Peak(s) L vs M (N={args.num_peaks})\n(MISO | Ref: TX)')
                ax3.set_xlabel('L')
                ax3.set_ylabel('M')

                obs_date = taxis_bins[0].strftime('%Y-%m-%d')
                fig.suptitle(f'{tx_name} to {rx_name} - {obs_date}\nMISO | Ref: TX', fontsize=16)
                for ax in [ax1, ax2]:
                    ax.set_ylim(range_min, range_max)
                fig.autofmt_xdate()
                plt.tight_layout()
                
                obs_datetime_str = taxis_bins[0].strftime("%Y%m%d_%H%M%S")
                plot_filename = os.path.join(gpath, f"correlate_tx_plot_{link_prefix}{obs_datetime_str}_{miso_suffix}_corr.png")
                plt.savefig(plot_filename)
                plt.close(fig)
                print(f"Saved plot: {plot_filename}")

    if summary_power is None and all_summed_power:
        summary_power = np.concatenate(all_summed_power, axis=-1)
        summary_doppler = np.concatenate(all_doppler, axis=-1)

    if summary_peak_L is None and all_peak_L:
        # Concatenate list of arrays (n_ranges, n_bins)
        summary_peak_L = np.concatenate(all_peak_L, axis=-1)
        summary_peak_M = np.concatenate(all_peak_M, axis=-1)
        summary_peak_P = np.concatenate(all_peak_P, axis=-1)

    if summary_peak_L2 is None and all_peak_L2:
        summary_peak_L2 = np.concatenate(all_peak_L2, axis=-1)
        summary_peak_M2 = np.concatenate(all_peak_M2, axis=-1)
        summary_peak_P2 = np.concatenate(all_peak_P2, axis=-1)

    # Accumulate peak data for geodetic plots and 1D Doppler plot
    all_lat_geo = []
    all_lon_geo = []
    all_alt_geo = []
    all_p_db_geo = []
    all_time_geo = []
    all_doppler_geo = []
    all_bragg_enu_geo = []
    all_range_geo = []
    all_peak_idx_geo = []

    if summary_peak_L is not None and summary_peak_M is not None:
        n_rng, n_bins_total = summary_peak_L.shape
        if len(all_taxis) == n_bins_total:
            summary_noise = np.median(summary_power)
            if summary_noise <= 0: summary_noise = 1e-12
            
            if global_tx_gps is not None and global_rx_gps is not None:
                for i in range(n_bins_total):
                    p_list = [(summary_peak_L[:, i], summary_peak_M[:, i], summary_peak_P[:, i])]
                    if args.num_peaks == 2 and summary_peak_L2 is not None:
                        p_list.append((summary_peak_L2[:, i], summary_peak_M2[:, i], summary_peak_P2[:, i]))
                    
                    # To store geo info for this bin
                    bin_geo = np.full((n_rng, 3), np.nan)
                    
                    for p_idx, (L_vals, M_vals, P_vals) in enumerate(p_list):
                        mask = ~np.isnan(L_vals) & ~np.isnan(M_vals)
                        if args.snr_summary_threshold is not None:
                            peak_snr = 10 * np.log10(np.maximum(P_vals / summary_noise, 1e-12))
                            mask &= (peak_snr >= args.snr_summary_threshold)
                        if not np.any(mask): continue
                        
                        ts_val = all_taxis[i].timestamp()
                        try:
                            lat_v, lon_v, alt_v, bragg_enu_v, _ = geolocation_from_bistatic_peak(
                                L_vals[mask], M_vals[mask], ranges[mask]*1000.0, global_tx_gps, global_rx_gps, wavelength
                            )
                            doppler_v = summary_doppler[:, i][mask] if summary_doppler is not None else np.zeros_like(lat_v)
                            geo_mask = (~np.isnan(lat_v)) & (alt_v/1000.0 >= geo_alt_min) & (alt_v/1000.0 <= geo_alt_max)
                            
                            if np.any(geo_mask):
                                all_lat_geo.extend(lat_v[geo_mask]); all_lon_geo.extend(lon_v[geo_mask]); all_alt_geo.extend(alt_v[geo_mask])
                                # Normalize noise by num_tx for SNR calculation
                                all_p_db_geo.extend(10 * np.log10(np.maximum(P_vals[mask][geo_mask]/(summary_noise*num_tx), 1e-12)))
                                all_time_geo.extend([ts_val] * np.sum(geo_mask)); all_doppler_geo.extend(doppler_v[geo_mask])
                                all_range_geo.extend(ranges[mask][geo_mask]*1000); all_peak_idx_geo.extend([p_idx + 1] * np.sum(geo_mask))
                                
                                # Store for summary points
                                bin_geo[mask, :] = np.stack([lat_v, lon_v, alt_v], axis=-1)
                        except: pass
                    if i % 500 == 0:
                        print(f"  Processed bin {i}/{n_bins_total}")

    # Summary Noise and SNR calculation
    summary_noise = np.median(summary_power)
    if summary_noise <= 0: summary_noise = 1e-12
    summary_snr = 10 * np.log10(np.maximum(summary_power / summary_noise, 1e-12))

    # Read theoretical Doppler if available
    theor_doppler_data = None
    theor_time_data = None
    theor_range_data = None
    theor_angle_data = None
    theor_h5_path = os.path.join(os.path.dirname(__file__), "misc_data", "theoretical_doppler_dt=0.0s.h5")
    if os.path.exists(theor_h5_path):
        try:
            with h5py.File(theor_h5_path, 'r') as f_theor:
                link_key = f"links/{link_name}"
                if link_key in f_theor:
                    theor_group = f_theor[link_key]
                    if 'doppler_hz' in theor_group:
                        theor_doppler_data = theor_group['doppler_hz'][:]
                        th_time = f_theor['time'][:]
                        # Convert unix timestamps to UTC datetime objects
                        theor_time_data = [datetime.fromtimestamp(t, timezone.utc) for t in th_time]
                        print(f"Loaded theoretical Doppler for link: {link_name}")
                    
                    if 'fitted_range_m' in theor_group:
                        theor_range_data = theor_group['fitted_range_m'][:] / 1000.0 # convert to km
                        print(f"Loaded theoretical range for link: {link_name}")
                    else:
                        theor_range_data = None

                    if 'bragg_angle_deg' in theor_group:
                        theor_angle_data = theor_group['bragg_angle_deg'][:]
                        print(f"Loaded theoretical Bragg angle for link: {link_name}")
                    else:
                        theor_angle_data = None
                else:
                    # Try case-insensitive or partial match if needed, but for now exact link_name
                    print(f"Note: Link {link_key} not found in theoretical Doppler file.")
                    theor_range_data = None
                    theor_angle_data = None
        except Exception as e:
            print(f"Warning: Could not read theoretical Doppler file: {e}")

    # RCS Estimation Setup
    radar_freq_hz = 0.5 * sc.c / wavelength # approx wavelength is 2-way? No, wavelength is from metadata.
    # Actually metadata 'wavelength' was loaded.
    radar_freq_hz = sc.c / wavelength
    tx_ecef = np.array(wgs84_lla_to_ecef(global_tx_gps[0], global_tx_gps[1], global_tx_gps[2]))
    rx_ecef = np.array(wgs84_lla_to_ecef(global_rx_gps[0], global_rx_gps[1], global_rx_gps[2]))

    def get_rcs(lin_snr, lats, lons, alts, ptx=500,brx=100.0, gtx=1.0):
        pts_ecef_tuple = wgs84_lla_to_ecef(lats, lons, alts)
        pts_ecef = np.stack(pts_ecef_tuple, axis=-1)
        # pts_ecef shape is (N, 3) or (3,)
        R_tx = np.sqrt(np.sum((pts_ecef - tx_ecef)**2, axis=-1))
        R_rx = np.sqrt(np.sum((pts_ecef - rx_ecef)**2, axis=-1))
        sn_n_n = lin_snr + 1.0 # (S+N)/N
        sigma = sn_plus_n_over_n_to_rcs(sn_n_n, R_tx, R_rx, frequency_hz=radar_freq_hz, P_tx=ptx, B_rx=brx, G_tx=gtx)
        return sigma

    # Prepare subplots in a 2x2 grid
    # Prepare subplots in a 2x2 grid with larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True, layout='tight')
    ax_snr_rti = axes[0, 0]
    ax_dop_rti = axes[0, 1]
    ax_snr_peak = axes[1, 0]
    ax_dop_comp = axes[1, 1]
    
    # List for uniform formatting
    axes_list = [ax_snr_rti, ax_dop_rti, ax_snr_peak, ax_dop_comp]
    
    extent_summary = [mdates.date2num(all_taxis[0]), mdates.date2num(all_taxis[-1]), ranges[0], ranges[-1]]
    
    # Power Subplot (Top Left)
    im1 = ax_snr_rti.imshow(summary_snr, aspect='auto', origin='lower', cmap=cmap_time, extent=extent_summary, vmin=snr_time_vmin, vmax=snr_time_vmax, interpolation='nearest')
    divider1 = make_axes_locatable(ax_snr_rti)
    cax1 = divider1.append_axes("right", size="5%", pad=0.08)
    cb1 = plt.colorbar(im1, cax=cax1)
    cb1.set_label('SNR (dB)', fontsize=16)
    cb1.ax.tick_params(labelsize=14)
    ax_snr_rti.set_title(f'SNR RTI: {global_tx_name} to {global_rx_name}\n({all_taxis[0].strftime("%Y-%m-%d")})', fontsize=20)
    ax_snr_rti.set_ylabel('Total Range (km)', labelpad=2, fontsize=16)
    ax_snr_rti.set_ylim(range_min, range_max)

    if theor_range_data is not None and theor_time_data is not None:
        ax_snr_rti.plot(theor_time_data, theor_range_data, '--', color='white', linewidth=1.5, label='Falcon 9 Optical Range')
        ax_snr_rti.legend(loc='upper right', fontsize='medium')

    if summary_doppler is not None:
        ax_dop_rti.set_facecolor('lightgray')
        im2 = ax_dop_rti.imshow(summary_doppler, aspect='auto', origin='lower', cmap='RdBu_r', extent=extent_summary, vmin=args.v_doppler_min, vmax=args.v_doppler_max, interpolation='nearest')
        divider2 = make_axes_locatable(ax_dop_rti)
        cax2 = divider2.append_axes("right", size="5%", pad=0.08)
        cb2 = plt.colorbar(im2, cax=cax2)
        cb2.set_label('Doppler Shift (Hz)', fontsize=16)
        cb2.ax.tick_params(labelsize=14)
        ax_dop_rti.set_title(f'Doppler RTI', fontsize=20)
        ax_dop_rti.set_ylabel('Total Range (km)', labelpad=2, fontsize=16)
        ax_dop_rti.set_ylim(range_min, range_max)
    else:
        ax_dop_rti.text(0.5, 0.5, 'No Doppler Data', ha='center', va='center', fontsize=20)
    
    # 1D SNR plot: Beamforming peaks vs Summary Power (all cells > threshold)
    rng_mask_snr = (ranges >= range_min) & (ranges <= range_max)
    
    # Use args.snr_summary_threshold if provided, otherwise fallback to snr_threshold
    plot_threshold = args.snr_summary_threshold if args.snr_summary_threshold is not None else snr_threshold
    
    # Extract subsets within range limits
    snr_sub = summary_snr[rng_mask_snr, :]
    
    # Find all points above threshold
    rng_idx, time_idx = np.where(snr_sub > plot_threshold)
    
    if len(time_idx) > 0 and len(all_lat_geo) > 0:
        t_arr = np.array(all_taxis)
        l_snr = 10**(snr_sub[rng_idx, time_idx]/10.0)
        
        # Find closest beamformed peak for each summary point
        # Using a KDTree for speed if there are many peaks
        geo_pts = np.stack([all_time_geo, all_range_geo], axis=-1)
        
        # query_pts: [time_seconds, range_meters]
        q_times = np.array([t.timestamp() for t in t_arr[time_idx]])
        query_pts = np.stack([q_times, ranges[rng_mask_snr][rng_idx]*1000.0], axis=-1)
        
        # Scale time and range to roughly compatible units for "closeness"
        std_t = np.std(all_time_geo) if len(all_time_geo) > 1 else 1.0
        std_r = np.std(all_range_geo) if len(all_range_geo) > 1 else 1.0
        
        tree_norm = spatial.KDTree(geo_pts / [std_t, std_r])
        _, idx_closest = tree_norm.query(query_pts / [std_t, std_r])
        
        c_lat = np.array(all_lat_geo)[idx_closest]
        c_lon = np.array(all_lon_geo)[idx_closest]
        c_alt = np.array(all_alt_geo)[idx_closest]
        
        summary_rcs = get_rcs(l_snr, c_lat, c_lon, c_alt, ptx=500,brx=100.0, gtx=1.0)
        summary_rcs_db = 10 * np.log10(np.maximum(summary_rcs, 1e-12))
        
        ax_snr_peak.scatter(t_arr[time_idx], summary_rcs_db, color='blue', s=80, alpha=0.5, label=f'Summary RCS (SNR > {plot_threshold} dB)', marker='*')
        
    elif len(time_idx) > 0:
        # Fallback to SNR if no geodetic data
        t_arr = np.array(all_taxis)
        ax_snr_peak.scatter(t_arr[time_idx], snr_sub[rng_idx, time_idx], color='blue', s=30, alpha=0.5, label=f'Summary SNR > {plot_threshold} dB', marker='.')

    if len(all_p_db_geo) > 0 and args.plot_peak_rcs:
        t_dt_geo = [datetime.fromtimestamp(t, timezone.utc) for t in all_time_geo]
        # Calculate RCS for beamforming peaks
        bm_snr_lin = 10**(np.array(all_p_db_geo)/10.0)
        bm_rcs = get_rcs(bm_snr_lin, np.array(all_lat_geo), np.array(all_lon_geo), np.array(all_alt_geo), ptx=500*num_tx, brx=100.0*num_tx, gtx=1.0) #*num_tx)
        bm_rcs_db = 10 * np.log10(np.maximum(bm_rcs, 1e-12))
        
        ax_snr_peak.scatter(t_dt_geo, bm_rcs_db, color='orange', s=50, alpha=0.5, label='Beamforming Peak RCS', marker='.')
    
    ax_snr_peak.set_title('Calculated RCS', fontsize=20)
    ax_snr_peak.set_ylabel('RCS (dBm$^2$)', fontsize=16)
    ax_snr_peak.legend(loc='upper left', fontsize='medium')
    ax_snr_peak.grid(True, linestyle=':', alpha=0.5)

    if theor_angle_data is not None and theor_time_data is not None:
        ax_snr_angle = ax_snr_peak.twinx()
        ax_snr_angle.plot(theor_time_data, theor_angle_data, 'm--', linewidth=1.5, label='Theoretical Bragg Angle')
        ax_snr_angle.set_ylabel('Aspect Angle (deg)', color='magenta', fontsize=16)
        ax_snr_angle.tick_params(axis='y', labelcolor='magenta', labelsize=14)
        # Combined legend could be tricky, for now separate or let them overlap
        # ax_snr_angle.legend(loc='upper right', fontsize='x-small')

    # Doppler comparison subplot
    if len(all_doppler_geo) > 0:
        t_dt_geo = [datetime.fromtimestamp(t, timezone.utc) for t in all_time_geo]
        ax_dop_comp.scatter(t_dt_geo, all_doppler_geo, color='blue', s=10, alpha=0.5, label='Measured Doppler', marker='.')
    
    if theor_doppler_data is not None:
        ax_dop_comp.plot(theor_time_data, theor_doppler_data, 'k--', linewidth=1.5, label='Expected Falcon 9 Doppler')
    elif len(all_doppler_geo) == 0:
        ax_dop_comp.text(0.5, 0.5, 'No Doppler Data', ha='center', va='center')
        
    ax_dop_comp.set_title('Doppler comparison', fontsize=20)
    ax_dop_comp.set_ylabel('Doppler (Hz)', fontsize=16)
    ax_dop_comp.legend(loc='upper right', fontsize='medium')
    ax_dop_comp.grid(True, linestyle=':', alpha=0.5)
#    ax_dop_comp.set_ylim(args.v_doppler_min, args.v_doppler_max)
    ax_dop_comp.set_ylim(-1500, 1500)


    # Formatting
    for ax in axes_list:
        ax.xaxis_date(tz=timezone.utc)
        
        # Determine time limits
        t_min_plot = extent_summary[0]
        t_max_plot = extent_summary[1]
        
        def parse_time(t_str, ref_dt):
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    tm = datetime.strptime(t_str, fmt)
                    return ref_dt.replace(hour=tm.hour, minute=tm.minute, second=tm.second, microsecond=0)
                except ValueError:
                    continue
            return None

        if all_taxis:
            if args.time_min:
                tm_dt = parse_time(args.time_min, all_taxis[0])
                if tm_dt:
                    t_min_plot = mdates.date2num(tm_dt)
                else:
                    print(f"Warning: Could not parse time_min '{args.time_min}'. Use HH:MM or HH:MM:SS.")
            
            if args.time_max:
                tm_dt = parse_time(args.time_max, all_taxis[0])
                if tm_dt:
                    t_max_plot = mdates.date2num(tm_dt)
                else:
                    print(f"Warning: Could not parse time_max '{args.time_max}'. Use HH:MM or HH:MM:SS.")
        
        ax.set_xlim(t_min_plot, t_max_plot)
        
        # Format X-axis based on duration
        duration_days = t_max_plot - t_min_plot
        duration_minutes = duration_days * 24 * 60
        if duration_minutes < 2:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
            
        ax.set_xmargin(0)
        ax.tick_params(axis='both', which='major', pad=2, labelsize=14)
    
    axes_list[-1].set_xlabel('Time (UTC)', labelpad=2, fontsize=16)
    axes_list[-2].set_xlabel('Time (UTC)', labelpad=2, fontsize=16)
    fig.autofmt_xdate()
    
    obs_date_str_actual = all_taxis[0].strftime("%Y%m%d")
    summary_filename = os.path.join(gpath, f"summary_time_power_doppler_plot_{link_prefix}{obs_date_str_actual}_{miso_suffix}_corr.png")
    plt.savefig(summary_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined summary plot: {summary_filename}")

    # Peak L/M Scatter Plot & Geolocation
    if len(all_lat_geo) > 0:
        all_lat_geo = np.array(all_lat_geo)
        all_lon_geo = np.array(all_lon_geo)
        all_alt_geo = np.array(all_alt_geo)
        all_p_db_geo = np.array(all_p_db_geo)
        all_time_geo = np.array(all_time_geo)
        all_doppler_geo = np.array(all_doppler_geo)
        all_bragg_enu_geo = np.array(all_bragg_enu_geo)
        all_range_geo = np.array(all_range_geo)
        all_peak_idx_geo = np.array(all_peak_idx_geo)
        
        # Map limits
        tx_lat, tx_lon = global_tx_gps[0], global_tx_gps[1]
        rx_lat, rx_lon = global_rx_gps[0], global_rx_gps[1]
        mid_lat = (tx_lat + rx_lat) / 2.0
        mid_lon = (tx_lon + rx_lon) / 2.0
        
        lat_lim_min, lat_lim_max = mid_lat - geo_delta_deg, mid_lat + geo_delta_deg
        lon_lim_min, lon_lim_max = mid_lon - geo_delta_deg, mid_lon + geo_delta_deg

        obs_date_str = all_taxis[0].strftime("%Y-%m-%d")
        obs_date_file = all_taxis[0].strftime("%Y%m%d")

        # --- Combined Global Summary Plots ---
        fig = plt.figure(figsize=(30, 10))
        
        # Subplot 1: Color by Time
        ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax1.coastlines(resolution='10m')
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        gl1.top_labels = False
        gl1.right_labels = False

        # Convert unix timestamps to UTC hours
        time_hours = np.array([(datetime.fromtimestamp(t, timezone.utc).hour + 
                       datetime.fromtimestamp(t, timezone.utc).minute/60.0 + 
                       datetime.fromtimestamp(t, timezone.utc).second/3600.0) for t in all_time_geo])
        
        mask_p1 = (all_peak_idx_geo == 1)
        mask_p2 = (all_peak_idx_geo == 2)
        
        if np.any(mask_p1):
            sc1 = ax1.scatter(all_lon_geo[mask_p1], all_lat_geo[mask_p1], c=time_hours[mask_p1], cmap='twilight', s=10, alpha=0.6, transform=ccrs.PlateCarree(), label='Peak 1')
            plt.colorbar(sc1, ax=ax1, label='Time (UTC Hour)', pad=0.08, shrink=0.7)
        if np.any(mask_p2):
            ax1.scatter(all_lon_geo[mask_p2], all_lat_geo[mask_p2], c='orange', s=10, alpha=0.6, transform=ccrs.PlateCarree(), label='Peak 2')
        
        # Plot TX and RX
        ax1.plot(tx_lon, tx_lat, 'r*', markersize=12, markeredgecolor='k', transform=ccrs.PlateCarree(), label=f'TX: {global_tx_name}')
        ax1.plot(rx_lon, rx_lat, 'b^', markersize=10, markeredgecolor='k', transform=ccrs.PlateCarree(), label=f'RX: {global_rx_name}')
        ax1.legend(loc='lower right', framealpha=0.8)
        
        ax1.set_extent([lon_lim_min, lon_lim_max, lat_lim_min, lat_lim_max], crs=ccrs.PlateCarree())
        ax1.set_title(f'Color: Time (UTC)\n{obs_date_str}')

        # Subplot 2: Color by Doppler
        ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax2.coastlines(resolution='10m')
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        gl2.top_labels = False
        gl2.right_labels = False

        if np.any(mask_p1):
            sc2 = ax2.scatter(all_lon_geo[mask_p1], all_lat_geo[mask_p1], c=all_doppler_geo[mask_p1], cmap='RdBu_r', s=10, alpha=0.6, transform=ccrs.PlateCarree(), vmin=args.v_doppler_min, vmax=args.v_doppler_max, label='Peak 1')
            plt.colorbar(sc2, ax=ax2, label='Doppler Shift (Hz)', pad=0.08, shrink=0.7)
        if np.any(mask_p2):
            ax2.scatter(all_lon_geo[mask_p2], all_lat_geo[mask_p2], c='orange', s=10, alpha=0.6, transform=ccrs.PlateCarree(), label='Peak 2')
        
        ax2.plot(tx_lon, tx_lat, 'r*', markersize=12, markeredgecolor='k', transform=ccrs.PlateCarree())
        ax2.plot(rx_lon, rx_lat, 'b^', markersize=10, markeredgecolor='k', transform=ccrs.PlateCarree())
        
        ax2.set_extent([lon_lim_min, lon_lim_max, lat_lim_min, lat_lim_max], crs=ccrs.PlateCarree())
        ax2.set_title(f'Color: Doppler Shift (Hz)\n{obs_date_str}')

        # Subplot 3: Altitude vs Longitude
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor('lightgray')
        
        if np.any(mask_p1):
            sc3 = ax3.scatter(all_lon_geo[mask_p1], all_alt_geo[mask_p1]/1000.0, c=all_doppler_geo[mask_p1], cmap='RdBu_r', s=10, alpha=0.6, vmin=args.v_doppler_min, vmax=args.v_doppler_max, label='Peak 1')
            plt.colorbar(sc3, ax=ax3, label='Doppler Shift (Hz)', pad=0.08, shrink=0.7)
        if np.any(mask_p2):
            ax3.scatter(all_lon_geo[mask_p2], all_alt_geo[mask_p2]/1000.0, c='orange', s=10, alpha=0.6, label='Peak 2')
        
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_xlim(lon_lim_min, lon_lim_max)
        ax3.set_ylim(geo_alt_min, geo_alt_max)
        ax3.grid(True, linestyle=':', alpha=0.5)
        ax3.set_title(f'Altitude vs Longitude (Color: Doppler)\n{obs_date_str}')
        
        fig.suptitle(f'Global Summary Geolocation: {global_tx_name} to {global_rx_name}\nMISO | Ref: TX', fontsize=20)
        
        fname_combined = os.path.join(gpath, f"summary_geolocation_combined_{link_prefix}{obs_date_file}_{miso_suffix}_corr.png")
        plt.savefig(fname_combined, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined global geolocation plot: {fname_combined}")

        # --- Range vs Time Plot with Parabola Fit ---
        fig_rt, ax_rt = plt.subplots(figsize=(12, 8))
        
        # Convert range to km
        range_km = all_range_geo / 1000.0
        
        # Plot points
        t_datetime_arr = np.array([datetime.fromtimestamp(t, timezone.utc) for t in all_time_geo])
        t_num = mdates.date2num(t_datetime_arr)
        
        if np.any(mask_p1):
            sc_rt = ax_rt.scatter(t_datetime_arr[mask_p1], range_km[mask_p1], c=all_p_db_geo[mask_p1], cmap='viridis', s=10, alpha=0.6, label='Peak 1')
            plt.colorbar(sc_rt, ax=ax_rt, label='Peak 1 Power (dB)')
        if np.any(mask_p2):
            ax_rt.scatter(t_datetime_arr[mask_p2], range_km[mask_p2], c='orange', s=10, alpha=0.6, label='Peak 2')
        
        # Fit parabola: Range = a*t^2 + b*t + c
        # Use a time offset to keep the fit stable
        t0 = np.min(all_time_geo)
        t_rel = all_time_geo - t0
        
        if len(t_rel) >= 3:
            poly_coeffs = np.polyfit(t_rel, range_km, 2)
            p_fit = np.poly1d(poly_coeffs)
            
            t_fit_rel = np.linspace(np.min(t_rel), np.max(t_rel), 100)
            range_fit = p_fit(t_fit_rel)
            t_fit_datetime = [datetime.fromtimestamp(t + t0, timezone.utc) for t in t_fit_rel]
            
            ax_rt.plot(t_fit_datetime, range_fit, 'r-', linewidth=2, label=f'Parabola Fit')
            print(f"  Parabola fit coefficients (t_offset={t0}): a={poly_coeffs[0]}, b={poly_coeffs[1]}, c={poly_coeffs[2]}")
        
        ax_rt.xaxis_date(tz=timezone.utc)
        ax_rt.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
        ax_rt.set_xlabel('Time (UTC)')
        ax_rt.set_ylabel('Total Range (km)')
        ax_rt.set_title(f'Range vs Time and Parabola Fit: {global_tx_name} to {global_rx_name}\n{obs_date_str}  [MISO | Ref: TX]')
        ax_rt.grid(True, linestyle=':', alpha=0.5)
        ax_rt.legend()
        fig_rt.autofmt_xdate()
        
        fname_rt = os.path.join(gpath, f"summary_range_time_fit_{link_prefix}{obs_date_file}_{miso_suffix}_corr.png")
        plt.savefig(fname_rt, bbox_inches='tight')
        plt.close(fig_rt)
        print(f"Saved Range-Time fit plot: {fname_rt}")

        # --- Save Geodetic Data to Separate HDF5 ---
        geodetic_h5_filename = os.path.join(geodetic_path, f"geodetic_data_{link_prefix}{obs_date_file}_{miso_suffix}_corr.h5")
        try:
            with h5py.File(geodetic_h5_filename, 'w') as gf:
                gf.create_dataset('latitude', data=all_lat_geo)
                gf.create_dataset('longitude', data=all_lon_geo)
                gf.create_dataset('altitude_m', data=all_alt_geo)
                gf.create_dataset('peak_power_db', data=all_p_db_geo)
                gf.create_dataset('range_m', data=all_range_geo)
                gf.create_dataset('time_unix', data=all_time_geo)
                gf.create_dataset('doppler_hz', data=all_doppler_geo)
                gf.create_dataset('peak_idx', data=all_peak_idx_geo)
                if len(all_bragg_enu_geo) > 0:
                    gf.create_dataset('bragg_enu', data=all_bragg_enu_geo)
                
                # Metadata
                gf.attrs['tx_name'] = global_tx_name
                gf.attrs['rx_name'] = global_rx_name
                gf.attrs['obs_date'] = obs_date_str
                gf.attrs['system'] = 'MISO'
                gf.attrs['reference_sensor'] = 'tx'
                if link_name:
                    gf.attrs['link_name'] = link_name
                if global_tx_gps is not None:
                    gf.attrs['tx_gps'] = global_tx_gps
                if global_rx_gps is not None:
                    gf.attrs['rx_gps'] = global_rx_gps
            print(f"Saved geodetic data to {geodetic_h5_filename}")
        except Exception as ge:
            print(f"Error saving geodetic HDF5: {ge}")
            
    # Save the resulting all_summed_power and all_taxis in a new hdf5 in directory gpath
    # ONLY save if we didn't just load it to avoid redundant writes/corruptions
    if not data_loaded_from_summary:
        summary_h5_filename = os.path.join(gpath, f"summary_data_{link_prefix}{obs_date_str_actual}_{miso_suffix}_corr.h5")
        try:
            with h5py.File(summary_h5_filename, 'w') as sf:
                sf.create_dataset('summary_power', data=summary_power)
                sf.create_dataset('summary_doppler', data=summary_doppler)
                # Convert datetime objects to unix timestamps (seconds from epoch)
                taxis_timestamps = np.array([t.timestamp() for t in all_taxis])
                sf.create_dataset('taxis', data=taxis_timestamps)
                sf.create_dataset('ranges', data=ranges)

                if all_taxis_spectra:
                    taxis_spectra_timestamps = np.array([t.timestamp() for t in all_taxis_spectra])
                    sf.create_dataset('taxis_spectra', data=taxis_spectra_timestamps)
                    
                if summary_peak_L is not None:
                    sf.create_dataset('summary_peak_L', data=summary_peak_L)
                if summary_peak_M is not None:
                    sf.create_dataset('summary_peak_M', data=summary_peak_M)
                if summary_peak_P is not None:
                    sf.create_dataset('summary_peak_P', data=summary_peak_P)
                    
                if summary_peak_L2 is not None:
                    sf.create_dataset('summary_peak_L2', data=summary_peak_L2)
                if summary_peak_M2 is not None:
                    sf.create_dataset('summary_peak_M2', data=summary_peak_M2)
                if summary_peak_P2 is not None:
                    sf.create_dataset('summary_peak_P2', data=summary_peak_P2)
                
                # Metadata attributes
                sf.attrs['tx_name'] = global_tx_name
                sf.attrs['rx_name'] = global_rx_name
                sf.attrs['obs_date'] = all_taxis[0].strftime("%Y-%m-%d")
                sf.attrs['system'] = 'MISO'
                sf.attrs['reference_sensor'] = 'tx'
                if link_name:
                    sf.attrs['link_name'] = link_name
                
                if global_tx_ant_coords is not None:
                    sf.attrs['tx_ant_coords'] = global_tx_ant_coords
                if global_tx_gps is not None:
                    sf.attrs['tx_gps'] = global_tx_gps
                if global_rx_gps is not None:
                    sf.attrs['rx_gps'] = global_rx_gps
                if wavelength is not None:
                    sf.attrs['wavelength'] = wavelength
                
            print(f"Saved summary data to {summary_h5_filename}")
        except Exception as se:
            print(f"Error saving summary HDF5: {se}")

    

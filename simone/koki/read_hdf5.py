import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.ndimage import uniform_filter1d, median_filter
import glob
import argparse
# import sys  # Unused
from mpl_toolkits.axes_grid1 import make_axes_locatable
from beamforming import beamform_coherences, detect_one_target_bartlet_clean_nlls, estimate_direction_cosines_nlls
from coordinates import geolocation_from_bistatic_peak
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

def read_hdf5_data(file_path, read_residuals=True, return_rx_coords=False):
    """
    Opens an HDF5 file and reads 'voltage', 'chunk_duration_ns', and 'chunk_start_time_ns' inside 'decoded_data'.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None, None, None, None, None, None

    try:
        with h5py.File(file_path, 'r') as f:
            if 'decoded_data' in f:
                group = f['decoded_data']
                if 'voltage' in group:
                    voltage_data = group['voltage'][:]
                    v_shape = voltage_data.shape
                    nrxs = v_shape[0]  # First dimension is receivers

                    # Read noise power metadata first for fallback and residuals
                    chunk_noise_power = 0.0
                    try:
                        if 'chunk_noise_power' in group:
                            chunk_noise_power = group['chunk_noise_power'][()]
                    except Exception as ne:
                        print(f"Warning: Could not read chunk_noise_power: {ne}")

                    # Read residual if it exists and requested
                    if read_residuals and 'residual' in group:
                        residual_data = group['residual'][:]
                        voltage_data = voltage_data + residual_data
                    else:
                        # If residuals are not read (or do not exist), add noise
                        if np.any(chunk_noise_power > 0):
                            # RMS = SQRT(chunk_noise / 2) for both Real and Imag parts
                            # total power = E[R^2 + I^2] = chunk_noise
                            rms = np.atleast_1d(np.sqrt(chunk_noise_power / 2.0))
                            
                            # Broadcast rms to voltage_data.shape
                            # v_shape is (nrxs, num_tx, num_ranges, n_samples)
                            if rms.size == nrxs:
                                rms_broadcast = rms[:, np.newaxis, np.newaxis, np.newaxis]
                            else:
                                rms_broadcast = rms[0]
                                
                            noise_real = np.random.normal(0, rms_broadcast, v_shape)
                            noise_imag = np.random.normal(0, rms_broadcast, v_shape)
                            voltage_data = voltage_data + (noise_real + 1j * noise_imag)
                        else:
                             print(f"Warning: No residuals read and chunk_noise_power is {chunk_noise_power}. No noise added.")
                    
                    # Read time metadata
                    start_time = group['chunk_start_time_ns'][()] if 'chunk_start_time_ns' in group else 0
                    duration = group['chunk_duration_ns'][()] if 'chunk_duration_ns' in group else 0
                    
                    # Read frequency from metadata/system/tx/frequency
                    wavelength = None
                    try:
                        if 'metadata' in f and 'system' in f['metadata'] and 'tx' in f['metadata/system'] and 'frequency' in f['metadata/system/tx']:
                            frequency = f['metadata/system/tx/frequency'][()]
                            wavelength = 299792458.0 / frequency
                    except Exception as fe:
                        print(f"Warning: Could not read frequency metadata: {fe}")

                    # Read tx and rx names
                    tx_name = "Unknown TX"
                    rx_name = "Unknown RX"
                    try:
                        if 'metadata/system/tx/name' in f:
                            tx_name = f['metadata/system/tx/name'][()].decode('utf-8') if isinstance(f['metadata/system/tx/name'][()], bytes) else f['metadata/system/tx/name'][()]
                        if 'metadata/system/rx/name' in f:
                            rx_name = f['metadata/system/rx/name'][()].decode('utf-8') if isinstance(f['metadata/system/rx/name'][()], bytes) else f['metadata/system/rx/name'][()]
                    except Exception as ne:
                        print(f"Warning: Could not read radar link names: {ne}")
                    # Read tx antenna coordinates and GPS
                    tx_ant_coords = None
                    rx_ant_coords = None
                    tx_gps = None
                    rx_gps = None
                    try:
                        if 'metadata/system/tx/antenna_coordinates' in f:
                            tx_ant_coords = f['metadata/system/tx/antenna_coordinates'][()]
                        if 'metadata/system/rx/antenna_coordinates' in f:
                            rx_ant_coords = f['metadata/system/rx/antenna_coordinates'][()]
                        if 'metadata/system/tx/gps' in f:
                            tx_gps = f['metadata/system/tx/gps'][()]
                        if 'metadata/system/rx/gps' in f:
                            rx_gps = f['metadata/system/rx/gps'][()]
                    except Exception as ge:
                        print(f"Warning: Could not read coordinates/GPS metadata: {ge}")

                    # Range offset: if receiver is "Bornim", add 15.0 km
                    range_offset = 0.0
                    if "bornim" in rx_name:
                        range_offset = 0.0  #-3.0 #-6.0 #3.0
                        print("Range offset is " + str(range_offset) + " km")


                    #Returning Conjugate of voltage to be consistent with other radars
                    if return_rx_coords:
                        return np.conj(voltage_data), start_time, duration, wavelength, tx_name, rx_name, tx_ant_coords, rx_ant_coords, tx_gps, rx_gps, chunk_noise_power, range_offset
                    else:
                        return np.conj(voltage_data), start_time, duration, wavelength, tx_name, rx_name, tx_ant_coords, tx_gps, rx_gps, chunk_noise_power, range_offset
                else:
                    print(f"Error: 'voltage' not found in 'decoded_data' in {file_path}")
            else:
                print(f"Error: 'decoded_data' group not found in {file_path}")
    except Exception as e:
        print(f"An error occurred reading {file_path}: {e}")
    
    if return_rx_coords:
        return None, None, None, None, None, None, None, None, None, None, 0.0, 0.0
    else:
        return None, None, None, None, None, None, None, None, None, 0.0, 0.0


# Examples of calls
# python read_hdf5.py --read_summary 0 --day 19 --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Eregion_2_5ms/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Eregion_2_5ms_figures/'
# python read_hdf5.py --read_summary 0 --day 19 --link_name "Jic_Anc"
# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_bornholm" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_figures/' --read_summary 1
# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_figures/' --read_summary 1
# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_figures/' --read_summary 1
# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "jruh_hagenow" --event "FALCON" --dpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_5ms/' --gpath '/Volumes/KCH_4TB_IAP/SIMONe/Falcon_event_5ms_figures/' --read_summary 1

# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_event_figures/' --read_residuals False --detect_one_nlls True --read_summary 0

# python read_hdf5.py  --year 2025 --month 2 --day 19 --link_name "kborn_hagenow" --event "FALCON" --dpath '/Users/jchau/junk/SIMONe/Falcon_event/' --gpath '/Users/jchau/junk/SIMONe/Falcon_spectral_figures/' --read_summary 0

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
    default_read_residuals = True
    
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
    parser.add_argument('--read_residuals', type=str2bool, default=default_read_residuals, help='Read residuals from HDF5 if they exist (True/False)')
    parser.add_argument('--detect_one_nlls', type=str2bool, default=False, help='Use NLLS refinement for single peak detection (True/False)')
    
    args = parser.parse_args()
    
    event = args.event
    link_name = args.link_name
    link_prefix = f"{link_name}_" if link_name else ""
    
    dpath = args.dpath
    delta_range = 3.00
    
    # Baseline/Default values
    snr_threshold = 1  # dB

    num_parts = 3           # How many parts of the data to process
    smooth_val_time = 10
    smooth_val_spectra = 200  #40
    median_filter_size = 5  # Median filter size along frequency axis
    snr_time_vmin = 0
    snr_time_vmax = 10
    snr_spec_vmin = 0
    snr_spec_vmax = 20
    geo_power_vmin = 40
    geo_power_vmax = 60
    geo_alt_min = 90  # km
    geo_alt_max = 110 # km
    range_min = 180
    range_max = 400         # 500
    v_min = -600
    v_max = 600
    lat_min_slice, lat_max_slice = -12.4, -12.2
    geo_delta_deg = 2.0

    # Event-specific overrides
    if event:
        if event.upper() == 'EEJ':
            print(f"Applying parameters for event: {event}")
            num_parts = 3           # How many parts of the data to process
            smooth_val_time = 10
            smooth_val_spectra = 200  #40
            median_filter_size = 5  # Median filter size along frequency axis
            snr_time_vmin = 0
            snr_time_vmax = 10
            snr_spec_vmin = 0
            snr_spec_vmax = 20
            geo_power_vmin = 40
            geo_power_vmax = 60
            geo_alt_min = 90  # km
            geo_alt_max = 110 # km
            range_min = 180
            range_max = 400         # 500
            lat_min_slice, lat_max_slice = -12.4, -12.2
            v_min = -600
            v_max = 600
            geo_delta_deg = 2.0    
        elif event.upper() == 'FALCON':
            print(f"Applying parameters for event: {event}")
            num_parts = 60 #3           # How many parts of the data to process
            smooth_val_time = 4 #10
            smooth_val_spectra = 2 #4 #200  #40
            median_filter_size = 0 #2 # Median filter size along frequency axis
            snr_time_vmin = -3
            snr_time_vmax = 20
            snr_spec_vmin = -3
            snr_spec_vmax = 20
            geo_power_vmin = 40
            geo_power_vmax = 60
            geo_alt_min = 40  # km
            geo_alt_max = 80 # km
            range_min = 100
            range_max = 600 
            v_min = -200 # -100
            v_max = 200 # 100
            lat_min_slice, lat_max_slice = 52, 54
            geo_delta_deg = 2.5
            snr_threshold = 3  # dB

        else:
            print(f"Warning: Unknown event '{event}'. Using default parameters.")
    cmap_time = 'viridis'
    cmap_spectra = 'gist_ncar'  #'inferno'
    cmap_coherence = 'terrain'
    ch_a = 0
    ch_b = 5
    # Use a moderate grid size in beamforming for speed
    grid_size_bf = 256  #64
    year = args.year
    month = args.month
    day = args.day
    date_str = f"{year:04d}{month:02d}{day:02d}"
    obs_date_str_actual = date_str
    
    gpath = args.gpath
    read_summary = args.read_summary
    summary_file_to_read = f'summary_data_{link_prefix}{date_str}.h5' # Specific file to read from gpath
    
    # Ensure the figures directory exists 
    if not os.path.exists(gpath):
        os.makedirs(gpath)
        print(f"Created directory: {gpath}")

    # Find h5 files in dpath for the given date
    file_list = sorted(glob.glob(os.path.join(dpath, f"*{link_prefix}*{date_str}*.h5")))
    print(f"Found {len(file_list)} files to process for {date_str}.")

    all_summed_power = []
    all_taxis = []

    # New accumulations for RGB bands
    all_summed_power_r = []
    all_summed_power_g = []
    all_summed_power_b = []
    all_taxis_spectra = []
    
    all_peak_L = []
    all_peak_M = []
    all_peak_P = []
    
    global_tx_name = ""
    global_rx_name = ""
    global_vel_axis = None
    wavelength = None
    global_rx_name = ""

    # Summary Plot and Data Loading
    summary_power = None
    summary_power_r = None
    summary_power_g = None
    summary_power_b = None
    summary_peak_L = None
    summary_peak_M = None
    summary_peak_P = None
    data_loaded_from_summary = False

    global_tx_ant_coords = None
    global_tx_gps = None
    global_rx_gps = None

    if read_summary:
        summary_h5_path = os.path.join(gpath, summary_file_to_read)
        if os.path.exists(summary_h5_path):
            print(f"\nReading summary from {summary_h5_path}...")
            try:
                with h5py.File(summary_h5_path, 'r') as sf:
                    summary_power = sf['summary_power'][:]
                    taxis_timestamps = sf['taxis'][:]
                    all_taxis = [datetime.fromtimestamp(t, timezone.utc) for t in taxis_timestamps]
                    
                    # Read RGB powers if they exist
                    if 'summary_power_r' in sf:
                        summary_power_r = sf['summary_power_r'][:]
                    if 'summary_power_g' in sf:
                        summary_power_g = sf['summary_power_g'][:]
                    if 'summary_power_b' in sf:
                        summary_power_b = sf['summary_power_b'][:]
                    
                    # Read spectra taxis if it exists, otherwise infer or ignore
                    if 'taxis_spectra' in sf:
                        taxis_spectra_timestamps = sf['taxis_spectra'][:]
                        all_taxis_spectra = [datetime.fromtimestamp(t, timezone.utc) for t in taxis_spectra_timestamps]
                         
                    # Read peak L/M if exist
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
                
                # Power
                power = np.abs(voltage)**2
                summed_power = np.sum(power, axis=(0, 1))
                
                # Spectra
                spectra = np.fft.fft(voltage, axis=-1)
                spectra_shifted = np.fft.fftshift(spectra, axes=-1)
                power_spectra = np.abs(spectra_shifted)**2
                summed_power_spectra = np.sum(power_spectra, axis=(0, 1))
                
                # Apply median filter along frequency axis (last axis)
                if median_filter_size >= 3:
                    summed_power_spectra = median_filter(summed_power_spectra, size=(1, median_filter_size))
                
                # Smoothing with uniform filter
                summed_power = uniform_filter1d(summed_power, size=smooth_val_time, axis=-1)
                summed_power_spectra = uniform_filter1d(summed_power_spectra, size=smooth_val_spectra, axis=-1)
                
                # Noise
                noise_time = np.median(summed_power)
                noise_spectra = np.median(summed_power_spectra)
                
    
                # Ensure they are not zero for division
                noise_time = max(noise_time, 1e-12)
                noise_spectra = max(noise_spectra, 1e-12)
                
                # Save for summary (decimate by smooth_val)
                all_summed_power.append(summed_power[:, ::smooth_val_time])
                all_taxis.extend(taxis[::smooth_val_time])
                
                # Doppler
                dt = (part_duration / 1e9) / num_samples
                freqs = np.fft.fftfreq(num_samples, d=dt)
                freqs_shifted = np.fft.fftshift(freqs)
                if wavelength:
                    vel_axis = freqs_shifted * (wavelength / 2.0)
                    x_label = 'Doppler Velocity (+ve towards) (m/s)'
                    x_extent = [vel_axis[0], vel_axis[-1]]
                else:
                    vel_axis = freqs_shifted / 1e3 # kHz
                    x_label = 'Frequency (kHz)'
                    x_extent = [freqs_shifted[0]/1e3, freqs_shifted[-1]/1e3]
                
                # Store decimated velocity axis for summary plot if not already stored
                if global_vel_axis is None:
                    # Decimate to match spectra decimation (same as smooth_val_spectra)
                    global_vel_axis = vel_axis[::smooth_val_spectra]

                # Cross Spectra
                num_channels = voltage.shape[1]
                summed_cross_spectra = np.zeros((num_channels, num_channels, num_ranges, num_samples), dtype=complex)
                for i in range(num_channels):
                    for j in range(num_channels):
                        summed_cross_spectra[i, j] = np.sum(spectra_shifted[:, i] * np.conj(spectra_shifted[:, j]), axis=0)
                
                # Apply median filter along frequency axis for cross-spectra
                if median_filter_size >= 3:
                    # Process real and imaginary parts separately
                    summed_cross_spectra_real = median_filter(summed_cross_spectra.real, size=(1, 1, 1, median_filter_size))
                    summed_cross_spectra_imag = median_filter(summed_cross_spectra.imag, size=(1, 1, 1, median_filter_size))
                    summed_cross_spectra = summed_cross_spectra_real + 1j * summed_cross_spectra_imag
                
                # Smoothing with uniform filter
                summed_cross_spectra = uniform_filter1d(summed_cross_spectra, size=smooth_val_spectra, axis=-1)

                # -------------------------------------------------------------------------
                # Beamforming Peak Estimation
                # -------------------------------------------------------------------------
                # Decimate spectra for beamforming to avoid redundant estimations
                summed_power_spectra_dec = summed_power_spectra[..., ::smooth_val_spectra]
                summed_cross_spectra_dec = summed_cross_spectra[..., ::smooth_val_spectra]
                snr_spectra_dec = 10 * np.log10(np.maximum(summed_power_spectra_dec / noise_spectra, 1e-12))
                
                num_samples_dec = summed_power_spectra_dec.shape[-1]
                peak_L = np.full((num_ranges, num_samples_dec), np.nan)
                peak_M = np.full((num_ranges, num_samples_dec), np.nan)
                peak_P = np.full((num_ranges, num_samples_dec), np.nan)
                
                # Check prerequisites: wavelength and antenna coords
                # tx_ant_coords should be (N_ants, 3)
                if wavelength is not None and global_tx_ant_coords is not None:
                    # 1. Identify valid points (SNR > threshold)
                    # snr_spectra_dec is (n_ranges, n_freqs_dec)
                    valid_mask = snr_spectra_dec > snr_threshold
                    
                    if np.any(valid_mask):
                        n_valid = np.sum(valid_mask)
                        print(f"  Running beamforming on {n_valid} points (decimated)...")
                        
                        # 2. Construct baselines and extract coherences
                        # Generate unique baselines indices (upper triangle)
                        n_ants = summed_cross_spectra_dec.shape[0] # num_channels
                        bl_vectors = []
                        bl_indices = []
                        
                        for i in range(n_ants):
                            for j in range(i + 1, n_ants):
                                bl = global_tx_ant_coords[i] - global_tx_ant_coords[j]
                                bl_vectors.append(bl)
                                bl_indices.append((i, j))
                                
                        if bl_vectors:
                            bl_vectors = np.array(bl_vectors)
                            
                            # Extract coherences for valid pixels
                            # coherences shape: (N_baselines, N_samples)
                            
                            batch_coherences = []
                            for (i, j) in bl_indices:
                                # Get (n_rng, n_freq_dec) slice
                                cs_slice = summed_cross_spectra_dec[i, j]
                                # Select valid pixels
                                vals = cs_slice[valid_mask]
                                batch_coherences.append(vals)
                                
                            batch_coherences = np.array(batch_coherences) # (N_bl, N_valid_points)
                            
                            # 3. Run Beamforming / Peak Detection in batches to save memory
                            bf_batch_size = 500
                            peaks_l = np.zeros(n_valid)
                            peaks_m = np.zeros(n_valid)
                            peaks_p = np.zeros(n_valid)

                            for b_start in range(0, n_valid, bf_batch_size):
                                b_end = min(b_start + bf_batch_size, n_valid)
                                current_batch = batch_coherences[:, b_start:b_end]
                                n_batch = b_end - b_start
                                
                                # Standard Bartlett beamforming as initial guess for this batch
                                L_grid, M_grid, P_batch = beamform_coherences(current_batch, bl_vectors, wavelength, grid_size=grid_size_bf)
                                
                                # 4. Find Peaks
                                P_flat = P_batch.reshape(n_batch, -1)
                                idx_max = np.nanargmax(P_flat, axis=1)
                                l_flat = L_grid.flatten()
                                m_flat = M_grid.flatten()
                                
                                if args.detect_one_nlls:
                                    # Refine each Bartlett peak individually
                                    print("Refine each Bartlett peak individually for NLLS")
                                    for idx_in_batch in range(n_batch):
                                        idx_global = b_start + idx_in_batch
                                        sample_coh = current_batch[:, idx_in_batch]
                                        l_init, m_init = l_flat[idx_max[idx_in_batch]], m_flat[idx_max[idx_in_batch]]
                                        
                                        l_est, m_est, a_est, succ_est = estimate_direction_cosines_nlls(
                                            sample_coh, bl_vectors, wavelength, initial_guess=(l_init, m_init)
                                        )
                                        peaks_l[idx_global] = l_est
                                        peaks_m[idx_global] = m_est
                                        peaks_p[idx_global] = np.abs(a_est)**2
                                else:
                                    peaks_l[b_start:b_end] = l_flat[idx_max]
                                    peaks_m[b_start:b_end] = m_flat[idx_max]
                                    peaks_p[b_start:b_end] = P_flat[np.arange(n_batch), idx_max]
                            
                            # 5. Store back
                            peak_L[valid_mask] = peaks_l
                            peak_M[valid_mask] = peaks_m
                            peak_P[valid_mask] = peaks_p
                
                # Already decimated, so append directly
                all_peak_L.append(peak_L)
                all_peak_M.append(peak_M)
                all_peak_P.append(peak_P)
                
                # Accumulate RGB powers

                # Accumulate RGB powers
                # summed_power_spectra is (n_ranges, n_freq_bins)
                n_freq = summed_power_spectra.shape[1]
                idx1 = n_freq // 3
                idx2 = 2 * (n_freq // 3)
                
                # Sum along frequency axis (axis 1) to get (n_ranges, 1)
                # We use keepdims=True so we can concatenate along time axis later
                p_r = np.sum(summed_power_spectra[:, :idx1], axis=1, keepdims=True)
                p_g = np.sum(summed_power_spectra[:, idx1:idx2], axis=1, keepdims=True)
                p_b = np.sum(summed_power_spectra[:, idx2:], axis=1, keepdims=True)
                
                all_summed_power_r.append(p_r)
                all_summed_power_g.append(p_g)
                all_summed_power_b.append(p_b)
                
                # Time axis for spectra (one point per chunk)
                all_taxis_spectra.append(taxis[len(taxis)//2])

                # Plotting
                fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
                (ax1, ax2), (ax3, ax4) = axs
                
                # Top Left: Time Power/Noise
                extent_time = [mdates.date2num(taxis[0]), mdates.date2num(taxis[-1]), ranges[0], ranges[-1]]
                snr_time = 10 * np.log10(np.maximum(summed_power / noise_time, 1e-12))
                im1 = ax1.imshow(snr_time, aspect='auto', origin='lower', cmap=cmap_time, extent=extent_time, vmin=snr_time_vmin, vmax=snr_time_vmax, interpolation='nearest')
                plt.colorbar(im1, ax=ax1, label='(dB)', pad=0.01)
                ax1.xaxis_date(tz=timezone.utc)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
                ax1.set_title('Time Power/Noise')
                ax1.set_ylabel('Total Range (km)')

                # Top Right: Power Spectra/Noise
                extent_freq = [x_extent[0], x_extent[1], ranges[0], ranges[-1]]
                snr_spectra = 10 * np.log10(np.maximum(summed_power_spectra / noise_spectra, 1e-12))
                im2 = ax2.imshow(snr_spectra, aspect='auto', origin='lower', cmap=cmap_spectra, extent=extent_freq, vmin=snr_spec_vmin, vmax=snr_spec_vmax, interpolation='nearest')
                plt.colorbar(im2, ax=ax2, label='(dB)', pad=0.01)
                ax2.set_title('Power Spectra/Noise')
                ax2.set_xlabel(x_label)
                if v_min is not None and v_max is not None:
                    ax2.set_xlim(v_min, v_max)

                # Bottom Left: Phase Cross Spectra
                phase_ab = np.angle(summed_cross_spectra[ch_a, ch_b])
                phase_ab_masked = np.where(snr_spectra < snr_threshold, np.nan, phase_ab)
                im3 = ax3.imshow(phase_ab_masked, aspect='auto', origin='lower', cmap='hsv', extent=extent_freq, interpolation='nearest')
                plt.colorbar(im3, ax=ax3, label='Phase (rad)', pad=0.01)
                ax3.set_title(f'Phase Cross Spectra (Ch {ch_a} & {ch_b})')
                ax3.set_xlabel(x_label)
                ax3.set_ylabel('Total Range (km)')
                if v_min is not None and v_max is not None:
                    ax3.set_xlim(v_min, v_max)

                # Bottom Right: Peak L/M Scatter
                # Re-using ax4 for scatter plot of peaks
                # Decimate vel_axis to match peak_L dimensions
                vel_axis_dec = vel_axis[::smooth_val_spectra]
                # peak_L is (num_ranges, num_freq_dec)
                # Tile velocity axis to match
                if peak_L.shape[1] == len(vel_axis_dec):
                    V_matrix = np.tile(vel_axis_dec, (num_ranges, 1))
                    
                    L_flat = peak_L.flatten()
                    M_flat = peak_M.flatten()
                    V_flat = V_matrix.flatten()
                    
                    # Mask NaNs
                    mask_valid = ~np.isnan(L_flat) & ~np.isnan(M_flat)
                    
                    if np.any(mask_valid):
                        vmin_sc = v_min if v_min is not None else x_extent[0]
                        vmax_sc = v_max if v_max is not None else x_extent[1]
                        sc = ax4.scatter(L_flat[mask_valid], M_flat[mask_valid], c=V_flat[mask_valid], 
                                       cmap='RdBu_r', s=5, alpha=0.5, marker='.', vmin=vmin_sc, vmax=vmax_sc)
                        
                        # Draw horizon circle
                        theta = np.linspace(0, 2*np.pi, 201)
                        ax4.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
                        
                        plt.colorbar(sc, ax=ax4, label='Doppler Velocity (m/s)' if wavelength else 'Frequency (kHz)')
                        ax4.set_xlim(-1, 1)
                        ax4.set_ylim(-1, 1)
                        ax4.set_aspect('equal')
                        ax4.grid(True, alpha=0.3)
                    else:
                        ax4.text(0.5, 0.5, 'No valid peaks', ha='center', va='center')
                else:
                    ax4.text(0.5, 0.5, 'Dim Mismatch', ha='center', va='center')
                    
                
                ax4.set_title(f'Peak L vs M (Ch {ch_a} & {ch_b})')
                ax4.set_xlabel('L')
                ax4.set_ylabel('M')

                obs_date = taxis[0].strftime('%Y-%m-%d')
                obs_time_str = taxis[0].strftime("%H%M%S")
                fig.suptitle(f'{tx_name} to {rx_name} - {obs_date}', fontsize=16)
                for ax in [ax1, ax2, ax3]:
                    ax.set_ylim(range_min, range_max)
                fig.autofmt_xdate()
                plt.tight_layout()
                
                # Filename includes date and time of the segment
                obs_datetime_str = taxis[0].strftime("%Y%m%d_%H%M%S")
                plot_filename = os.path.join(gpath, f"power_plot_{link_prefix}{obs_datetime_str}.png")
                # Remove bbox_inches='tight' to ensure consistent image size (determined by figsize)
                plt.savefig(plot_filename) #, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot: {plot_filename}")

    if summary_power is None and all_summed_power:
        summary_power = np.concatenate(all_summed_power, axis=-1)

    if summary_power_r is None and all_summed_power_r:
        summary_power_r = np.concatenate(all_summed_power_r, axis=-1)
        summary_power_g = np.concatenate(all_summed_power_g, axis=-1)
        summary_power_b = np.concatenate(all_summed_power_b, axis=-1)
        
    if summary_peak_L is None and all_peak_L:
        # Concatenate list of arrays (n_ranges, n_timesteps)
        summary_peak_L = np.concatenate(all_peak_L, axis=-1)
        summary_peak_M = np.concatenate(all_peak_M, axis=-1)
        summary_peak_P = np.concatenate(all_peak_P, axis=-1)

    if summary_power is not None:
        print("\nCreating summary time power plot...")
        summary_noise = np.median(summary_power)
        if summary_noise <= 0:
            summary_noise = 1e-12
        summary_snr = 10 * np.log10(np.maximum(summary_power / summary_noise, 1e-12))
        
        fig, ax = plt.subplots(figsize=(12, 6), layout='tight')
        extent_summary = [mdates.date2num(all_taxis[0]), mdates.date2num(all_taxis[-1]), ranges[0], ranges[-1]]
        im = ax.imshow(summary_snr, aspect='auto', origin='lower', cmap=cmap_time, extent=extent_summary, vmin=snr_time_vmin, vmax=snr_time_vmax, interpolation='nearest')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.02)
        plt.colorbar(im, cax=cax, label='(dB)')
        
        ax.xaxis_date(tz=timezone.utc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
        title_prefix = f"[{link_name}] " if link_name else ""
        ax.set_title(f'{title_prefix}Summary Time Power/Noise: {global_tx_name} to {global_rx_name} - {all_taxis[0].strftime("%Y-%m-%d")}')
        ax.set_xlabel('Time (UTC)', labelpad=2)
        ax.set_ylabel('Total Range (km)', labelpad=2)
        ax.set_ylim(range_min, range_max)
        ax.set_xlim(extent_summary[0], extent_summary[1])
        ax.set_xmargin(0)
        ax.tick_params(axis='both', which='major', pad=2)
        fig.autofmt_xdate()
        
        obs_date_str_actual = all_taxis[0].strftime("%Y%m%d")
        summary_filename = os.path.join(gpath, f"summary_time_power_plot_{link_prefix}{obs_date_str_actual}.png")
        plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0)
        print(f"Saved summary plot: {summary_filename}")

        # RGB Summary Plot
        if summary_power_r is not None and summary_power_g is not None and summary_power_b is not None:
            print("\nCreating summary RGB power plot...")
            # Normalize each channel
            # Assuming similar noise floor for all, or estimate per channel
            noise_r = np.median(summary_power_r)
            noise_g = np.median(summary_power_g)
            noise_b = np.median(summary_power_b)
            
            # Helper to normalize to [0, 1] for RGB
            def normalize_snr(p, noise_val, vmin, vmax):
                if noise_val <= 0:
                    noise_val = 1e-12
                snr = 10 * np.log10(np.maximum(p / noise_val, 1e-12))
                norm = (snr - vmin) / (vmax - vmin)
                return np.clip(norm, 0, 1)
            
            # Using same vmin/vmax for all channels for now, or define specific ones
            rgb_vmin = snr_spec_vmin # Use spectra limits or time limits? User didn't specify. Spectra limits seem appropriate since derived from spectra.
            rgb_vmax = snr_spec_vmax 
            
            # Note: summary_power_r shape is (n_ranges, n_chunks)
            # Need to create RGB image (n_ranges, n_chunks, 3)
            # Transpose to (n_ranges, n_chunks) if needed, but they are already that way.
            
            r_norm = normalize_snr(summary_power_r, noise_r, rgb_vmin, rgb_vmax/2)
            g_norm = normalize_snr(summary_power_g, noise_g, rgb_vmin, rgb_vmax)
            b_norm = normalize_snr(summary_power_b, noise_b, rgb_vmin, rgb_vmax/2)
            
            rgb_img = np.dstack((r_norm, g_norm, b_norm))
            
            fig_rgb, ax_rgb = plt.subplots(figsize=(12, 6), layout='tight')
            
            # Time axis for spectra
            # Use all_taxis_spectra if available, else try to use start/end of all_taxis
            if all_taxis_spectra:
                 date_start = all_taxis_spectra[0]
                 date_end = all_taxis_spectra[-1]
            else:
                 date_start = all_taxis[0]
                 date_end = all_taxis[-1]

            extent_rgb = [mdates.date2num(date_start), mdates.date2num(date_end), ranges[0], ranges[-1]]
            
            im_rgb = ax_rgb.imshow(rgb_img, aspect='auto', origin='lower', extent=extent_rgb, interpolation='nearest')
            
            ax_rgb.xaxis_date(tz=timezone.utc)
            ax_rgb.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
            
            ax_rgb.set_title(f'Summary RGB Power (R:Low, G:Mid, B:High): {global_tx_name} to {global_rx_name} - {date_start.strftime("%Y-%m-%d")}')
            ax_rgb.set_xlabel('Time (UTC)', labelpad=2)
            ax_rgb.set_ylabel('Total Range (km)', labelpad=2)
            ax_rgb.set_ylim(range_min, range_max)
            ax_rgb.set_xlim(extent_rgb[0], extent_rgb[1])
            ax_rgb.set_xmargin(0)
            ax_rgb.tick_params(axis='both', which='major', pad=2)

            # Add white space to the right (dummy axis to match colorbar width of the other plot)
            divider = make_axes_locatable(ax_rgb)
            cax = divider.append_axes("right", size="2%", pad=0.02)
            cax.axis('off')

            fig_rgb.autofmt_xdate()
            
            summary_rgb_filename = os.path.join(gpath, f"summary_rgb_power_plot_{link_prefix}{obs_date_str_actual}.png")
            plt.savefig(summary_rgb_filename, bbox_inches='tight', pad_inches=0.25)
            print(f"Saved RGB summary plot: {summary_rgb_filename}")
            
        # Peak L/M Scatter Plot & Geolocation
        if summary_peak_L is not None and summary_peak_M is not None and global_vel_axis is not None:
            print("\nCreating summary Peak L/M scatter plots and Geolocation maps per time step...")
            
            # Create subdirectory
            sc_path = os.path.join(gpath, "summary_scatter_plots")
            os.makedirs(sc_path, exist_ok=True)
            
            n_rng, n_cols = summary_peak_L.shape
            n_freq = len(global_vel_axis)
            
            if n_cols % n_freq == 0:
                n_chunks = n_cols // n_freq
                
                # Check if we have timestamps for these chunks
                if all_taxis_spectra and len(all_taxis_spectra) == n_chunks:
                    print(f"Generating {n_chunks} sets of plots in {sc_path}...")
                    
                    if global_tx_gps is None or global_rx_gps is None:
                        print("Warning: TX or RX GPS coordinates missing. Generating standard scatter plots (L vs M) only.")
                    else:
                        print("TX/RX GPS coordinates found. Generating Geolocation maps.")

                    # Determine Label
                    cbar_label = 'Doppler Velocity/Frequency'
                    if np.max(np.abs(global_vel_axis)) < 5000:
                        cbar_label = 'Doppler Velocity (m/s)'
                    else:
                        cbar_label = 'Frequency (kHz)'

                    if v_min is not None and v_max is not None:
                        v_lim = max(abs(v_min), abs(v_max))
                    else:
                        v_lim = np.max(np.abs(global_vel_axis))
                    for i in range(n_chunks):
                        # Extract chunk data
                        col_start = i * n_freq
                        col_end = (i + 1) * n_freq
                        
                        L_chunk = summary_peak_L[:, col_start:col_end] # (n_rng, n_freq)
                        M_chunk = summary_peak_M[:, col_start:col_end]
                        P_chunk = summary_peak_P[:, col_start:col_end]
                        
                        # Velocity matrix (same for every chunk)
                        V_chunk = np.tile(global_vel_axis, (n_rng, 1))
                        
                        # Range matrix (tiled for frequencies)
                        # ranges is (n_rng,)
                        R_chunk = np.tile(ranges[:, np.newaxis], (1, n_freq))
                        # Convert ranges to meters (assuming they are in km in the file)
                        R_chunk_m = R_chunk * 1000.0
                        
                        # Flatten
                        L_flat = L_chunk.flatten()
                        M_flat = M_chunk.flatten()
                        V_flat = V_chunk.flatten()
                        R_flat = R_chunk_m.flatten()
                        P_flat = P_chunk.flatten()
                        
                        # Filter NaNs
                        mask = ~np.isnan(L_flat) & ~np.isnan(M_flat)
                        L_valid = L_flat[mask]
                        M_valid = M_flat[mask]
                        V_valid = V_flat[mask]
                        R_valid = R_flat[mask]
                        P_valid = P_flat[mask]
                        
                        if len(L_valid) > 0:
                            ts = all_taxis_spectra[i]
                            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                            file_ts = ts.strftime("%Y%m%d_%H%M%S")
                            
                            # Geolocation Plots (if GPS data available)
                            if global_tx_gps is not None and global_rx_gps is not None:
                                # Calculate coordinates
                                try:
                                    # If wavelength is available, we get 5 returns
                                    if wavelength is not None:
                                        lat_valid, lon_valid, alt_valid, k_bragg_valid, bragg_lambda_valid = geolocation_from_bistatic_peak(
                                            L_valid, M_valid, R_valid, global_tx_gps, global_rx_gps, wavelength
                                        )
                                    else:
                                        lat_valid, lon_valid, alt_valid = geolocation_from_bistatic_peak(
                                            L_valid, M_valid, R_valid, global_tx_gps, global_rx_gps
                                        )
                                        k_bragg_valid = None
                                        bragg_lambda_valid = None
                                    
                                    # Filter out failed geolocations (NaNs) and altitude limits
                                    geo_mask = (~np.isnan(lat_valid)) & (alt_valid/1000.0 >= geo_alt_min) & (alt_valid/1000.0 <= geo_alt_max)
                                    
                                    if np.any(geo_mask):
                                        lat_geo = lat_valid[geo_mask]
                                        lon_geo = lon_valid[geo_mask]
                                        alt_geo = alt_valid[geo_mask]
                                        V_geo = V_valid[geo_mask]
                                        P_geo = P_valid[geo_mask]
                                        # Calculate power in dB
                                        P_db_geo = 10 * np.log10(np.maximum(P_geo, 1e-12))
                                        
                                        # Calculate map limits centered on midpoint
                                        tx_lat, tx_lon = global_tx_gps[0], global_tx_gps[1]
                                        rx_lat, rx_lon = global_rx_gps[0], global_rx_gps[1]
                                        mid_lat = (tx_lat + rx_lat) / 2.0
                                        mid_lon = (tx_lon + rx_lon) / 2.0
                                        
                                       
                                        lat_lim_min = mid_lat - geo_delta_deg
                                        lat_lim_max = mid_lat + geo_delta_deg
                                        lon_lim_min = mid_lon - geo_delta_deg
                                        lon_lim_max = mid_lon + geo_delta_deg
                                        
                                        

                                        # Combined Plot: Lon-Lat (Vel), Lon-Lat (Alt), Lon-Lat (Power), Right=Alt vs Lon slice
                                        fig_geo = plt.figure(figsize=(18, 14))
                                        ax1 = fig_geo.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
                                        ax2 = fig_geo.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
                                        ax3 = fig_geo.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
                                        ax4_slice = fig_geo.add_subplot(2, 2, 4)
                                        
                                        # Add map features to map axes
                                        for ax_map in [ax1, ax2, ax3]:
                                            ax_map.coastlines(resolution='10m')
                                            ax_map.add_feature(cfeature.BORDERS, linestyle=':')
                                            ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                                            # Force map axes to fill the subplot area to match ax4_slice
                                            ax_map.set_aspect('auto', adjustable='datalim')
                                            # Gridlines with labels
                                            gl = ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
                                            gl.top_labels = False
                                            gl.right_labels = False

                                        # Plot 1: Velocity
                                        sc1 = ax1.scatter(lon_geo, lat_geo, c=V_geo, cmap='RdBu_r', s=10, alpha=0.7, marker='o', vmin=-v_lim, vmax=v_lim, transform=ccrs.PlateCarree())
                                        plt.colorbar(sc1, ax=ax1, label=cbar_label)
                                        ax1.plot(tx_lon, tx_lat, 'r^', markersize=10, label='TX', transform=ccrs.PlateCarree())
                                        ax1.plot(rx_lon, rx_lat, 'gv', markersize=10, label='RX', transform=ccrs.PlateCarree())
                                        # Draw slice limits
                                        ax1.plot([lon_lim_min, lon_lim_max], [lat_min_slice, lat_min_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax1.plot([lon_lim_min, lon_lim_max], [lat_max_slice, lat_max_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax1.legend(loc='lower left')
                                        ax1.set_title(f'Color: {cbar_label}')
                                        ax1.set_extent([lon_lim_min, lon_lim_max, lat_lim_min, lat_lim_max], crs=ccrs.PlateCarree())
                                        
                                        # Plot 2: Altitude
                                        sc2 = ax2.scatter(lon_geo, lat_geo, c=alt_geo/1000.0, cmap='terrain', s=10, alpha=0.7, marker='o', vmin=geo_alt_min, vmax=geo_alt_max, transform=ccrs.PlateCarree())
                                        plt.colorbar(sc2, ax=ax2, label='Altitude (km)')
                                        ax2.plot(tx_lon, tx_lat, 'r^', markersize=10, label='TX', transform=ccrs.PlateCarree())
                                        ax2.plot(rx_lon, rx_lat, 'gv', markersize=10, label='RX', transform=ccrs.PlateCarree())
                                        # Draw slice limits
                                        ax2.plot([lon_lim_min, lon_lim_max], [lat_min_slice, lat_min_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax2.plot([lon_lim_min, lon_lim_max], [lat_max_slice, lat_max_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax2.legend(loc='lower left')
                                        ax2.set_title(f'Color: Altitude (km) [{geo_alt_min}-{geo_alt_max}]')
                                        ax2.set_extent([lon_lim_min, lon_lim_max, lat_lim_min, lat_lim_max], crs=ccrs.PlateCarree())

                                        # Plot 3: Power (dB)
                                        sc3 = ax3.scatter(lon_geo, lat_geo, c=P_db_geo, cmap='magma', s=10, alpha=0.7, marker='o', vmin=geo_power_vmin, vmax=geo_power_vmax, transform=ccrs.PlateCarree())
                                        plt.colorbar(sc3, ax=ax3, label='Peak Power (dB)')
                                        ax3.plot(tx_lon, tx_lat, 'r^', markersize=10, label='TX', transform=ccrs.PlateCarree())
                                        ax3.plot(rx_lon, rx_lat, 'gv', markersize=10, label='RX', transform=ccrs.PlateCarree())
                                        ax3.plot([lon_lim_min, lon_lim_max], [lat_min_slice, lat_min_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax3.plot([lon_lim_min, lon_lim_max], [lat_max_slice, lat_max_slice], color='k', linestyle='--', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
                                        ax3.legend(loc='lower left')
                                        ax3.set_title(f'Color: Peak Power (dB) [{geo_power_vmin}-{geo_power_vmax}]')
                                        ax3.set_extent([lon_lim_min, lon_lim_max, lat_lim_min, lat_lim_max], crs=ccrs.PlateCarree())

                                        # Plot 4: Altitude vs Longitude (Lat slice)
                                        mask_lat = (lat_geo >= lat_min_slice) & (lat_geo <= lat_max_slice)
                                        
                                        # Match X-axis formatting with the map plots
                                        ax4_slice.xaxis.set_major_formatter(cticker.LongitudeFormatter())
                                        ax4_slice.set_xlim(lon_lim_min, lon_lim_max)
                                        ax4_slice.set_ylim(geo_alt_min, geo_alt_max)
                                        ax4_slice.set_facecolor('lightgray')
                                        ax4_slice.grid(True, alpha=0.3, color='white')
                                        
                                        if np.any(mask_lat):
                                            sc4 = ax4_slice.scatter(lon_geo[mask_lat], alt_geo[mask_lat]/1000.0, c=V_geo[mask_lat], cmap='RdBu_r', s=10, alpha=0.7, marker='o', vmin=-v_lim, vmax=v_lim)
                                            plt.colorbar(sc4, ax=ax4_slice, label=cbar_label)
                                            ax4_slice.set_title( f'Alt vs Lon (Lat: {lat_min_slice} to {lat_max_slice})')
                                            ax4_slice.set_xlabel('Longitude')
                                            ax4_slice.set_ylabel('Altitude (km)')
                                        else:
                                            ax4_slice.text(0.5, 0.5, f'No data in range\n[{lat_min_slice}, {lat_max_slice}]', ha='center', va='center', transform=ax4_slice.transAxes)
                                        
                                        fig_geo.suptitle(f'Geolocation Summary: {global_tx_name} to {global_rx_name}\n{ts_str}', fontsize=16)
                                        fig_geo.tight_layout()
                                        
                                        fname_geo = os.path.join(sc_path, f"peak_geolocation_{link_prefix}{file_ts}.png")
                                        plt.savefig(fname_geo, bbox_inches='tight')
                                        plt.close(fig_geo)
                                except Exception as e:
                                    print(f"Error in geolocation calculation for frame {i}: {e}")
                            
                            else:
                                # Fallback: Standard L vs M Plot
                                fig_sc, ax_sc = plt.subplots(figsize=(8, 8))
                                sc = ax_sc.scatter(L_valid, M_valid, c=V_valid, cmap='RdBu_r', s=10, alpha=0.7, marker='o', vmin=-v_lim, vmax=v_lim)
                                theta = np.linspace(0, 2*np.pi, 201)
                                ax_sc.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
                                plt.colorbar(sc, ax=ax_sc, label=cbar_label)
                                
                                ax_sc.set_title(f'Peak L vs M: {global_tx_name} to {global_rx_name}\n{ts_str}')
                                ax_sc.set_xlabel('L (East-West)')
                                ax_sc.set_ylabel('M (North-South)')
                                ax_sc.set_xlim(-1, 1)
                                ax_sc.set_ylim(-1, 1)
                                ax_sc.set_aspect('equal')
                                ax_sc.grid(True, alpha=0.3)
                                fname = os.path.join(sc_path, f"peak_scatter_{link_prefix}{file_ts}.png")
                                plt.savefig(fname, bbox_inches='tight')
                                plt.close(fig_sc)
                                
                            if i % 10 == 0:
                                print(f"  Processed frame {i}/{n_chunks}")
                        else:
                            # print(f"  Frame {i}: No valid data.")
                            pass
                    print(f"Finished generating plots.")

                else:
                    print(f"Error: Number of chunks ({n_chunks}) does not match timestamp list ({len(all_taxis_spectra) if all_taxis_spectra else 0}).")
            else:
                print(f"Warning: Summary columns ({n_cols}) not divisible by frequency axis len ({n_freq}). Skipping scatter.")
            
        # Save the resulting all_summed_power and all_taxis in a new hdf5 in directory gpath
        # ONLY save if we didn't just load it to avoid redundant writes/corruptions
        if not data_loaded_from_summary:
            summary_h5_filename = os.path.join(gpath, f"summary_data_{link_prefix}{obs_date_str_actual}.h5")
            try:
                with h5py.File(summary_h5_filename, 'w') as sf:
                    sf.create_dataset('summary_power', data=summary_power)
                    # Convert datetime objects to unix timestamps (seconds from epoch)
                    taxis_timestamps = np.array([t.timestamp() for t in all_taxis])
                    sf.create_dataset('taxis', data=taxis_timestamps)
                    sf.create_dataset('ranges', data=ranges)

                    # Save RGB summaries
                    if summary_power_r is not None:
                        sf.create_dataset('summary_power_r', data=summary_power_r)
                    if summary_power_g is not None:
                        sf.create_dataset('summary_power_g', data=summary_power_g)
                    if summary_power_b is not None:
                        sf.create_dataset('summary_power_b', data=summary_power_b)
                    
                    if all_taxis_spectra:
                        taxis_spectra_timestamps = np.array([t.timestamp() for t in all_taxis_spectra])
                        sf.create_dataset('taxis_spectra', data=taxis_spectra_timestamps)
                        
                    if summary_peak_L is not None:
                        sf.create_dataset('summary_peak_L', data=summary_peak_L)
                    if summary_peak_M is not None:
                        sf.create_dataset('summary_peak_M', data=summary_peak_M)
                    if summary_peak_P is not None:
                        sf.create_dataset('summary_peak_P', data=summary_peak_P)
                    
                    if global_vel_axis is not None:
                        sf.create_dataset('vel_axis', data=global_vel_axis)
                    
                    # Metadata attributes
                    sf.attrs['tx_name'] = global_tx_name
                    sf.attrs['rx_name'] = global_rx_name
                    sf.attrs['obs_date'] = all_taxis[0].strftime("%Y-%m-%d")
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

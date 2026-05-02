import numpy as np

# WGS84 Constants
A = 6378137.0          # Semi-major axis (meters)
F_INV = 298.257223563  # Inverse flattening
F = 1.0 / F_INV
B = A * (1.0 - F)      # Semi-minor axis (meters)
E2 = F * (2 - F)       # Eccentricity squared

def wgs84_lla_to_ecef(lat, lon, alt):
    """
    Convert Geodetic coordinates (Lat, Lon, Alt) to ECEF (x, y, z).
    Lat, Lon in degrees. Alt in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    N = A / np.sqrt(1 - E2 * sin_lat**2)
    
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - E2) + alt) * sin_lat
    
    return x, y, z

def wgs84_ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates (x, y, z) to Geodetic (Lat, Lon, Alt).
    Returns Lat, Lon in degrees, Alt in meters.
    Uses Ferrari's solution or an iterative approach. Here using simple iterative.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)
    
    # Initial approximation
    lon = np.arctan2(y, x)
    
    e_prime_sq = (A**2 - B**2) / B**2
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    lat = np.arctan2(z + e_prime_sq * B * sin_theta**3, p - E2 * A * cos_theta**3)
    
    # Iterate for better precision (usually 1-2 iterations enough)
    # Using the closed form approximation is often sufficient, but let's do standard Heiskanen-Moritz
    
    # Recalculate N using initial lat
    sin_lat = np.sin(lat)
    N = A / np.sqrt(1 - E2 * sin_lat**2)
    alt = p / np.cos(lat) - N
    
    # Refine
    # For high precision apps, iterating is good, but for typical radar ranges this is very close.
    # Let's clean up return
    return np.degrees(lat), np.degrees(lon), alt

def enu_to_ecef_rotation(lat, lon):
    """
    Returns rotation matrix to convert ENU vector to ECEF vector.
    v_ecef = R @ v_enu
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Row vectors of R
    # R[:, 0] = E axis in ECEF = [-sin_lon, cos_lon, 0]
    # R[:, 1] = N axis in ECEF = [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
    # R[:, 2] = U axis in ECEF = [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    
    R = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon,  -sin_lat * sin_lon, cos_lat * sin_lon],
        [0,         cos_lat,           sin_lat]
    ])
    
    return R

def ecef_to_enu_rotation(lat, lon):
    """
    Returns rotation matrix to convert ECEF vector to ENU vector.
    v_enu = R @ v_ecef
    R is proper orthogonal, so R_inv = R_transpose.
    """
    # The matrix from enu_to_ecef_rotation is R_enu2ecef.
    # We want R_ecef2enu = R_enu2ecef.T
    return enu_to_ecef_rotation(lat, lon).T

def geolocation_from_bistatic_peak(L, M, total_range, tx_coords, rx_coords, wavelength=None):
    """
    Calculate target Geodetic coordinates from bistatic radar measurements.

    Parameters:
    - L, M: Direction cosines (East, North) with respect to the Transmitter.
    - total_range: Total propagation path length (TX -> Target -> RX) in meters.
    - tx_coords: (Lat, Lon, Alt) of Transmitter.
    - rx_coords: (Lat, Lon, Alt) of Receiver.
    - wavelength: (Optional) Radar wavelength in meters. If provided, returns Bragg wavelength.

    Returns:
    - target_lat, target_lon, target_alt
    - (Optional) bragg_vector_enu, bragg_wavelength (if wavelength is provided)
    """
    # 1. Transmitter and Receiver ECEF positions
    tx_pos = np.array(wgs84_lla_to_ecef(*tx_coords))
    rx_pos = np.array(wgs84_lla_to_ecef(*rx_coords))
    
    # Baseline vector (TX to RX)
    B_vec = rx_pos - tx_pos
    baseline_len = np.linalg.norm(B_vec)
    
    # 2. Local direction vector at TX
    # Assume upper hemisphere N > 0
    # Inputs L, M could be arrays
    L = np.asarray(L)
    M = np.asarray(M)
    total_range = np.asarray(total_range)
    
    # Broadcast shapes to common shape
    # This handles cases where total_range might match L/M in shape, or be singular per range bin, etc.
    L, M, total_range = np.broadcast_arrays(L, M, total_range)
    
    # Valid mask for physical directions
    R2 = L**2 + M**2
    # Ensure non-negative before sqrt
    valid_dirs = R2 <= 1.0
    N = np.zeros_like(L)
    N[valid_dirs] = np.sqrt(1.0 - R2[valid_dirs])
    # If R2 > 1, N=0 or NaN. Let's keep 0 and handle validity later.
    
    # Local ENU vector u_local = [L, M, N]
    # Shape handling for arrays
    u_local = np.stack((L, M, N), axis=-1) # (..., 3)
    
    # 3. Rotate to ECEF
    # Get rotation matrix for TX location
    R_enu2ecef = enu_to_ecef_rotation(tx_coords[0], tx_coords[1])
    
    # u_ecef = R @ u_local
    # tensor contraction: (..., 3) . (3, 3) -> (..., 3)
    # R is (3,3). We want u_ecef[i] = R @ u_local[i]
    u_ecef = u_local @ R_enu2ecef.T
    
    # 4. Solve for Range from TX (R_tx)
    # Formula: R_tx = (R_total^2 - |B|^2) / (2 * (R_total - u_ecef . B))
    
    u_dot_B = np.einsum('...i,i->...', u_ecef, B_vec)
    
    numerator = total_range**2 - baseline_len**2
    denominator = 2 * (total_range - u_dot_B)
    
    # Avoid division by zero
    R_tx = np.full_like(numerator, np.nan)
    mask = (denominator != 0) & valid_dirs
    
    R_tx[mask] = numerator[mask] / denominator[mask]
    
    # 5. Target Position in ECEF
    # P_target = P_tx + R_tx * u_ecef
    target_pos = tx_pos + R_tx[..., np.newaxis] * u_ecef
    
    # Handle Bragg computation if wavelength is provided
    bragg_lambda = None
    if wavelength is not None:
        # k_i = u_ecef (Unit direction TX -> Target)
        
        # Calculate k_s (Unit direction Target -> RX)
        # Vector Target -> RX
        vec_tgt_rx = rx_pos - target_pos
        dist_tgt_rx = np.linalg.norm(vec_tgt_rx, axis=-1)
        
        # Avoid div by zero
        k_s = np.zeros_like(vec_tgt_rx)
        valid_dist = dist_tgt_rx > 0
        k_s[valid_dist] = (2*np.pi/wavelength)*vec_tgt_rx[valid_dist] / dist_tgt_rx[valid_dist, np.newaxis]
        
        # Bragg vector (normalized magnitude): |k_s - k_i| (Scattering vector difference)
        # Often defined as k_diff = k_s - k_i assuming k vectors point along propagation
        # Note: In backscatter k_s = -k_i, so diff is -2k_i, mag is 2.
        # Bragg vector = k_s - k_i= 2*np.pi/Lambda_Bragg
        
        # Using incident direction unit vectors
        # k_incident = u_ecef (TX -> Tgt)
        # k_scattered = - k_s (if k_s is Tgt->RX, then scattered wave direction is Tgt->RX. Yes.
        # Physics: Delta k = k_out - k_in = k_scat - k_inc.
        # directions are u_ecef (TX->Tgt) and k_s (Tgt->RX).
        # So Delta_k_norm = |k_s - u_ecef|.
        
        k_i = (2*np.pi/wavelength)*u_ecef

        k_diff = k_s - k_i
        k_diff_mag = np.linalg.norm(k_diff, axis=-1)
        
        bragg_lambda = np.full_like(R_tx, np.nan)
        valid_bragg = (k_diff_mag > 1e-6) & mask # Avoid singularity
        bragg_lambda[valid_bragg] = 2*np.pi / k_diff_mag[valid_bragg]

    # 6. Convert to Geodetic
    # We apply wgs84_ecef_to_lla to the last axis
    # Flatten if needed or apply along axis
    if target_pos.ndim == 1:
        lat, lon, alt = wgs84_ecef_to_lla(target_pos[0], target_pos[1], target_pos[2])
    else:
        # For arrays, explicit loop or vectorization:
        # wgs84_ecef_to_lla handles numpy arrays for x, y, z components
        lat, lon, alt = wgs84_ecef_to_lla(
            target_pos[..., 0], target_pos[..., 1], target_pos[..., 2]
        )
    
    if wavelength is not None:
        # Rotate Bragg vector to Local ENU at Target
        # k_diff is in ECEF (shape: ..., 3)
        
        # Calculate rotation matrices for each target point
        # This is expensive for large arrays if done in python loop. 
        # But for vectorized numpy, we can construct the matrix.
        
        # lat, lon are degrees arrays matching the shape
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # R_ecef2enu components (based on transpose of earlier R)
        # Row 0 (East): [-sin_lon, cos_lon, 0]
        # Row 1 (North): [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
        # Row 2 (Up): [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
        
        # Construct Bragg ENU components
        # k_diff = (k_x, k_y, k_z)
        kx = k_diff[..., 0]
        ky = k_diff[..., 1]
        kz = k_diff[..., 2]
        
        # East
        b_e = -sin_lon * kx + cos_lon * ky
        
        # North
        b_n = -sin_lat * cos_lon * kx - sin_lat * sin_lon * ky + cos_lat * kz
        
        # Up
        b_u = cos_lat * cos_lon * kx + cos_lat * sin_lon * ky + sin_lat * kz
        
        bragg_vector_enu = np.stack((b_e, b_n, b_u), axis=-1)
        
        return lat, lon, alt, bragg_vector_enu, bragg_lambda
    else:
        return lat, lon, alt

if __name__ == "__main__":
    # Test
    # TX at Jicamarca
    tx = (-11.95, -76.87, 500)
    # RX nearby
    rx = (-12.96, -76.86, 500)
    
    wavelength = 10.0 # meters

    L = [-0.2,0.0,0.2]
    M = [-0.05,-0.05,-0.05]
    total_range = [300e3,300e3,300e3] # 300 km
    
    lat, lon, alt, k_bragg_enu, bragg_lambda = geolocation_from_bistatic_peak(L, M, total_range, tx, rx, wavelength)
    print(f"Target Location: Lat {lat}, Lon {lon}, Alt {alt} m")
    print(f"Bragg Vector (ENU): {k_bragg_enu}")
    print(f"Bragg Wavelength: {bragg_lambda} m")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def estimate_direction_cosines_nlls(coherences, baselines, wavelength, initial_guess=(0.0, 0.0)):
    """
    Estimates direction cosines (l, m) from complex correlations using 
    non-linear least squares (NLLS).
    
    Fits for (l, m, A_real, A_imag) where A is the complex amplitude.
    
    Parameters:
    - coherences: array-like of complex numbers (N_baselines,)
    - baselines: array-like (N_baselines, 3) in meters
    - wavelength: float, radar wavelength in meters
    - initial_guess: tuple (l_init, m_init)
    
    Returns:
    - l, m: Estimated direction cosines
    - complex_amplitude: complex number
    - success: bool
    """
    coherences = np.array(coherences)
    baselines = np.array(baselines)
    k = 2 * np.pi / wavelength
    
    # params: [l, m, A_real, A_imag]
    def residuals(params):
        l, m, ar, ai = params
        r2 = l**2 + m**2
        if r2 > 1.0:
            # Penalty for going out of bounds
            return np.ones(2 * len(coherences)) * 1e6 * (r2 - 1.0)
        
        n = np.sqrt(1.0 - r2)
        complex_amp = ar + 1j * ai
        
        # Phase delay: k * (b . s)
        phases = k * (baselines[:, 0] * l + baselines[:, 1] * m + baselines[:, 2] * n)
        model = complex_amp * np.exp(1j * phases)
        
        res = coherences - model
        return np.concatenate([res.real, res.imag])

    # Initial amplitude guess: mean magnitude
    a_init = np.mean(np.abs(coherences))
    x0 = [initial_guess[0], initial_guess[1], a_init, 0.0]
    
    # Bounds for l and m
    lower_bounds = [-1.0, -1.0, -np.inf, -np.inf]
    upper_bounds = [1.0, 1.0, np.inf, np.inf]
    
    try:
        res = least_squares(residuals, x0, bounds=(lower_bounds, upper_bounds))
        l_est, m_est, ar_est, ai_est = res.x
        return l_est, m_est, ar_est + 1j * ai_est, res.success
    except Exception as e:
        print(f"NLLS Error: {e}")
        return initial_guess[0], initial_guess[1], 0j, False

def detect_one_target_bartlet_clean_nlls(coherences, baselines, wavelength, grid_size=400):
    """
    Detects one target using Bartlett beamforming and refines it with NLLS.
    
    1. Runs Bartlett beamforming to find the peak (initial guess).
    2. Refines (l, m, amplitude) using single-target NLLS.
    
    Returns:
    - (l_est, m_est): Refined direction cosines.
    - complex_amplitude: Refined complex amplitude.
    - success: bool
    """
    # 1. Beamforming to find peak
    L, M, P = beamform_coherences(coherences, baselines, wavelength, grid_size=grid_size)
    idx = np.nanargmax(P)
    l_init, m_init = L.flatten()[idx], M.flatten()[idx]
    
    # 2. NLLS refinement
    return estimate_direction_cosines_nlls(coherences, baselines, wavelength, initial_guess=(l_init, m_init))
def estimate_direction_cosines_nlls_two_targets(coherences, baselines, wavelength, initial_guess1=(0.0, 0.0), initial_guess2=(0.0, 0.0)):
    """
    Jointly estimates direction cosines for two targets using non-linear least squares.
    
    Fits for (l1, m1, ar1, ai1, l2, m2, ar2, ai2).
    
    Parameters:
    - coherences: (N_baselines,) complex visibility data
    - baselines: (N_baselines, 3) geometry
    - wavelength: radar wavelength (m)
    - initial_guess1: (l1, m1)
    - initial_guess2: (l2, m2)
    
    Returns:
    - (l1, m1), (l2, m2): Estimated direction cosines
    - amp1, amp2: Estimated complex amplitudes
    - success: bool
    """
    coherences = np.array(coherences)
    baselines = np.array(baselines)
    k = 2 * np.pi / wavelength
    n_bl = len(coherences)

    # params: [l1, m1, ar1, ai1, l2, m2, ar2, ai2]
    def residuals(params):
        l1, m1, ar1, ai1, l2, m2, ar2, ai2 = params
        
        # Check unit circle constraint for both
        r2_1 = l1**2 + m1**2
        r2_2 = l2**2 + m2**2
        
        penalty = 0.0
        if r2_1 > 1.0:
            penalty += (r2_1 - 1.0)
        if r2_2 > 1.0:
            penalty += (r2_2 - 1.0)
            
        if penalty > 0:
            return np.ones(2 * n_bl) * 1e6 * penalty

        n1 = np.sqrt(1.0 - r2_1)
        n2 = np.sqrt(1.0 - r2_2)
        
        amp1 = ar1 + 1j * ai1
        amp2 = ar2 + 1j * ai2
        
        phases1 = k * (baselines[:, 0] * l1 + baselines[:, 1] * m1 + baselines[:, 2] * n1)
        phases2 = k * (baselines[:, 0] * l2 + baselines[:, 1] * m2 + baselines[:, 2] * n2)
        
        model = amp1 * np.exp(1j * phases1) + amp2 * np.exp(1j * phases2)
        res = coherences - model
        return np.concatenate([res.real, res.imag])

    # Improved initial amplitude guess via Projection
    def get_complex_amp(l, m):
        n = np.sqrt(max(0.0, 1.0 - l**2 - m**2))
        phases = k * (baselines[:, 0] * l + baselines[:, 1] * m + baselines[:, 2] * n)
        return np.mean(coherences * np.exp(-1j * phases))

    amp1_init = get_complex_amp(*initial_guess1)
    amp2_init = get_complex_amp(*initial_guess2)

    x0 = [initial_guess1[0], initial_guess1[1], amp1_init.real, amp1_init.imag,
          initial_guess2[0], initial_guess2[1], amp2_init.real, amp2_init.imag]
    
    # Bounds
    low = [-1.0, -1.0, -np.inf, -np.inf, -1.0, -1.0, -np.inf, -np.inf]
    high = [1.0, 1.0, np.inf, np.inf, 1.0, 1.0, np.inf, np.inf]
    
    try:
        res = least_squares(residuals, x0, bounds=(low, high))
        p = res.x
        return (p[0], p[1]), (p[4], p[5]), (p[2] + 1j * p[3]), (p[6] + 1j * p[7]), res.success
    except Exception as e:
        print(f"Two-target NLLS Error: {e}")
        return initial_guess1, initial_guess2, 0j, 0j, False


def beamform_clean_two_peaks(coherences, baselines, wavelength, grid_size=400, clean_gain=1.0):
    """
    Finds the first and second peak of the spatial power spectrum using a 
    CLEAN-like procedure.
    
    1. Performs beamforming to find the primary source (L1, M1).
    2. Estimating the complex amplitude A1 of the primary source.
    3. Subtracts the primary source contribution from the input coherences:
       V_res = V_meas - gain * A1 * exp(j * k * b . s1)
    4. Performs beamforming on the residual coherences to find the second source (L2, M2).
    
    Parameters:
    - coherences: (N_baselines,) complex visibility data
    - baselines: (N_baselines, 3) geometry
    - wavelength: radar wavelength (m)
    - grid_size: resolution of the sky search
    - clean_gain: amount of first peak to remove (0.0 to 1.0)
    
    Returns:
    - (l1, m1), (l2, m2): Estimated direction cosines
    - a1: Estimated complex amplitude of the first peak
    - P1: First power map
    - P2: Residual power map
    """
    # 1. Find first peak
    L, M, P1 = beamform_coherences(coherences, baselines, wavelength, grid_size=grid_size)
    
    idx1 = np.nanargmax(P1)
    l1, m1 = L.flatten()[idx1], M.flatten()[idx1]
    
    # 2. Estimate amplitude A1
    k = 2 * np.pi / wavelength
    r2_1 = l1**2 + m1**2
    n1 = np.sqrt(max(1.0 - r2_1, 0.0))
    s1 = np.array([l1, m1, n1])
    
    # Phase delay: k * (b . s)
    phases1 = k * (baselines @ s1)
    # A = mean( V * exp(-j * phase) )
    a1 = np.mean(coherences * np.exp(-1j * phases1))
    
    # 3. Clean subtraction
    coherences_res = coherences - clean_gain * a1 * np.exp(1j * phases1)
    
    # 4. Find second peak from residual
    _, _, P2 = beamform_coherences(coherences_res, baselines, wavelength, grid_size=grid_size)
    idx2 = np.nanargmax(P2)
    l2, m2 = L.flatten()[idx2], M.flatten()[idx2]
    
    return (l1, m1), (l2, m2), a1, P2

def beamform_capon(covariance_matrix, antennas, wavelength, grid_size=200, epsilon=1e-3):
    """
    Capon's Beamforming (Minimum Variance Distortionless Response - MVDR).
    Provides higher resolution than Bartlett (Direct Fourier) by adaptively 
    minimizing interference while maintaining gain towards the steering direction.

    P(s) = 1 / (a(s)^H * R_inv * a(s))

    Parameters:
    - covariance_matrix: (N_ant, N_ant) complex covariance matrix R.
    - antennas: (N_ant, 3) antenna positions in meters.
    - wavelength: wavelength in meters.
    - grid_size: resolution of the direction cosine grid.
    - epsilon: diagonal loading factor for regularization (e.g., 1e-3).

    Returns:
    - L, M: Grid of direction cosines.
    - P: Capon power spectrum.
    """
    R = np.array(covariance_matrix)
    antennas = np.array(antennas)
    n_ant = R.shape[0]

    # 1. Regularization and Inversion
    # R_inv = inv(R + epsilon * Trace(R)/N * I)
    loading = epsilon * np.trace(R).real / n_ant
    R_inv = np.linalg.pinv(R + loading * np.eye(n_ant))

    # 2. Grid Setup
    l = np.linspace(-1, 1, grid_size)
    m = np.linspace(-1, 1, grid_size)
    L, M = np.meshgrid(l, m)
    mask = (L**2 + M**2) <= 1.0
    
    L_v = L[mask]
    M_v = M[mask]
    N_v = np.sqrt(1.0 - L_v**2 - M_v**2)
    s_vectors = np.vstack((L_v, M_v, N_v)) # (3, N_pix)

    # 3. Compute Steering Vectors and Capon Power
    k = 2 * np.pi / wavelength
    # phase: (N_ant, N_pix)
    phase = k * (antennas @ s_vectors)
    A = np.exp(1j * phase) # (N_ant, N_pix)

    # den = diag(A.H * R_inv * A)
    # We can compute this efficiently: Sum_i Sum_j A*[i, p] * R_inv[i, j] * A[j, p]
    R_inv_A = R_inv @ A # (N_ant, N_pix)
    den = np.sum(np.conj(A) * R_inv_A, axis=0) # (N_pix,)
    
    P_v = 1.0 / np.abs(den)

    # 4. Reshape
    P = np.full((grid_size, grid_size), np.nan)
    P[mask] = P_v
    
    return L, M, P

def beamform_capon_batch(coherences, antennas, wavelength, grid_size=200, epsilon=1e-3):
    """
    Batched version of Capon beamforming for multiple samples.
    """
    coherences = np.array(coherences)
    antennas = np.array(antennas)
    n_ant = len(antennas)
    n_batch = coherences.shape[1]
    
    # 1. Build R matrices: (N_batch, N_ant, N_ant)
    R = np.zeros((n_batch, n_ant, n_ant), dtype=complex)
    for b in range(n_batch):
        R[b] = build_covariance_matrix(coherences[:, b], n_ant)
        
    # 2. Regularization and Inversion
    # loading: (N_batch,)
    loading = epsilon * np.trace(R, axis1=1, axis2=2).real / n_ant
    R_loaded = R + loading[:, np.newaxis, np.newaxis] * np.eye(n_ant)
    R_inv = np.linalg.pinv(R_loaded) # (N_batch, N_ant, N_ant)
    
    # 3. Grid Setup
    l = np.linspace(-1, 1, grid_size)
    m = np.linspace(-1, 1, grid_size)
    L, M = np.meshgrid(l, m)
    mask = (L**2 + M**2) <= 1.0
    
    L_v = L[mask]
    M_v = M[mask]
    N_v = np.sqrt(np.maximum(0.0, 1.0 - L_v**2 - M_v**2))
    s_vectors = np.vstack((L_v, M_v, N_v)) # (3, N_pix)
    
    k = 2 * np.pi / wavelength
    phase = k * (antennas @ s_vectors)
    A = np.exp(1j * phase) # (N_ant, N_pix)
    
    # 4. Compute Denominator: Sum_i Sum_j A*[i, p] * R_inv[b, i, j] * A[j, p]
    # R_inv: (N_batch, N_ant, N_ant), A: (N_ant, N_pix) -> (N_batch, N_ant, N_pix)
    R_inv_A = R_inv @ A 
    
    # den: (N_batch, N_pix) = Sum_ant ( conj(A[ant, p]) * R_inv_A[b, ant, p] )
    den = np.sum(np.conj(A) * R_inv_A, axis=1) # (N_batch, N_pix)
    
    P_v = 1.0 / np.abs(den) # (N_batch, N_pix)
    
    P_batch = np.full((n_batch, grid_size, grid_size), np.nan)
    P_batch[:, mask] = P_v
    
    return L, M, P_batch

def detect_two_targets_capon_clean_batch(coherences, antennas, wavelength, grid_size=400, clean_gain=1.0):
    """
    Batched version of two-target detection using Capon-CLEAN.
    """
    coherences = np.array(coherences)
    n_batch = coherences.shape[1]
    baselines = np.array(list(baselines_from_antennas(antennas)))
    k = 2 * np.pi / wavelength
    
    # 1. Peak 1 with Capon
    L, M, P1_batch = beamform_capon_batch(coherences, antennas, wavelength, grid_size=grid_size)
    
    P1_flat = P1_batch.reshape(n_batch, -1)
    idx1 = np.nanargmax(P1_flat, axis=1)
    l1 = L.flatten()[idx1]
    m1 = M.flatten()[idx1]
    
    # 2. Estimate Amplitudes A1
    n1 = np.sqrt(np.maximum(0.0, 1.0 - l1**2 - m1**2))
    s1 = np.vstack((l1, m1, n1)) # (3, N_batch)
    
    # Phase delays for s1 for each baseline: (N_bl, N_batch)
    # phases1[bl, b] = k * (baselines[bl] @ s1[:, b])
    phases1 = k * (baselines @ s1) # (N_bl, N_batch)
    
    # a1[b] = mean( coherences[:, b] * exp(-1j * phases1[:, b]) )
    a1 = np.mean(coherences * np.exp(-1j * phases1), axis=0)
    
    # 3. Residual Coherences
    coherences_res = coherences - clean_gain * a1 * np.exp(1j * phases1)
    
    # 4. Peak 2 with Capon on Residuals
    _, _, P2_batch = beamform_capon_batch(coherences_res, antennas, wavelength, grid_size=grid_size)
    
    P2_flat = P2_batch.reshape(n_batch, -1)
    idx2 = np.nanargmax(P2_flat, axis=1)
    l2 = L.flatten()[idx2]
    m2 = M.flatten()[idx2]
    
    return (l1, m1), (l2, m2)

def build_covariance_matrix(coherences, n_ant):
    """Reconstructs the covariance matrix R from a list of complex coherences."""
    R = np.eye(n_ant, dtype=complex)
    idx = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            if idx < len(coherences):
                R[i, j] = coherences[idx]
                R[j, i] = np.conj(coherences[idx])
                idx += 1
    return R

def detect_two_targets_capon_clean_nlls(coherences, antennas, wavelength, grid_size=400, clean_gain=1.0):
    """
    Finds two targets using a Capon-CLEAN approach followed by Joint NLLS.
    
    1. Reconstructs R from coherences.
    2. Runs Capon beamforming to find Peak 1.
    3. Estimates the complex amplitude of Peak 1.
    4. Subtracts Peak 1 contribution from coherences to get residuals.
    5. Reconstructs R_res from residual coherences.
    6. Runs Capon beamforming on R_res to find Peak 2.
    7. Refines both peaks using Joint NLLS.
    
    Returns:
    - (l1, m1), (l2, m2): Joint NLLS refined positions.
    - amp1, amp2: Joint NLLS refined complex amplitudes.
    - success: bool
    """
    coherences = np.array(coherences)
    antennas = np.array(antennas)
    n_ant = len(antennas)
    k = 2 * np.pi / wavelength
    
    # 1. First Peak with Capon
    R1 = build_covariance_matrix(coherences, n_ant)
    L, M, P1 = beamform_capon(R1, antennas, wavelength, grid_size=grid_size)
    
    idx1 = np.nanargmax(P1)
    l1_c, m1_c = L.flatten()[idx1], M.flatten()[idx1]
    
    # 2. Estimate amplitude A1
    n1 = np.sqrt(max(0.0, 1.0 - l1_c**2 - m1_c**2))
    s1 = np.array([l1_c, m1_c, n1])
    
    # Compute steering vector for s1
    baselines = np.array(list(baselines_from_antennas(antennas)))
    phases1 = k * (baselines @ s1)
    a1_est = np.mean(coherences * np.exp(-1j * phases1))
    
    # 3. Residual coherences (CLEAN subtraction)
    coherences_res = coherences - clean_gain * a1_est * np.exp(1j * phases1)
    
    # 4. Second Peak with Capon on residual
    R2 = build_covariance_matrix(coherences_res, n_ant)
    _, _, P2 = beamform_capon(R2, antennas, wavelength, grid_size=grid_size)
    
    idx2 = np.nanargmax(P2)
    l2_c, m2_c = L.flatten()[idx2], M.flatten()[idx2]
    
    print(f"Capon-CLEAN Initial Peaks: T1=({l1_c:.4f}, {m1_c:.4f}), T2=({l2_c:.4f}, {m2_c:.4f})")
    
    # 5. Joint NLLS refinement
    return estimate_direction_cosines_nlls_two_targets(coherences, baselines, 
                                                    wavelength, (l1_c, m1_c), (l2_c, m2_c))

def detect_two_targets_capon_nlls(covariance_matrix, coherences, antennas, wavelength, grid_size=400, mask_radius=0.1):
    """
    Finds two targets by combining Capon beamforming and Joint NLLS refinement (Masking approach).
    """
    # 1. Capon Beamforming
    L, M, P = beamform_capon(covariance_matrix, antennas, wavelength, grid_size=grid_size)
    
    # 2. Find first peak
    idx1 = np.nanargmax(P)
    l1_c, m1_c = L.flatten()[idx1], M.flatten()[idx1]
    
    # 3. Mask around first peak to find second peak
    dist_sq = (L - l1_c)**2 + (M - m1_c)**2
    P_masked = P.copy()
    P_masked[dist_sq < mask_radius**2] = np.nan
    
    idx2 = np.nanargmax(P_masked)
    l2_c, m2_c = L.flatten()[idx2], M.flatten()[idx2]
    
    print(f"Capon Masking Initial Peaks: T1=({l1_c:.4f}, {m1_c:.4f}), T2=({l2_c:.4f}, {m2_c:.4f})")
    
    baselines = np.array(list(baselines_from_antennas(antennas)))
    # 4. Joint NLLS refinement
    return estimate_direction_cosines_nlls_two_targets(coherences, baselines, 
                                                    wavelength, (l1_c, m1_c), (l2_c, m2_c))
def baselines_from_antennas(antennas):
    """Helper to generate baselines in the same order as simulate_coherences."""
    n_ant = len(antennas)
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            yield antennas[i] - antennas[j]

def beamform_coherences(coherences, baselines, wavelength, grid_size=200):
    """
    Performs Bartlett beamforming (Direct Fourier Transform) based on complex coherences 
    and baseline vectors.

    This function computes the spatial power spectrum over the entire sky (direction cosines).
    It assumes the 'z' component of the direction vector is positive (upper hemisphere).

    Parameters:
    - coherences: array-like of complex numbers. The complex visibility/coherence for each baseline.
                  Shape: (N_baselines,)
    - baselines: array-like. The difference vectors (dx, dy, dz) for each baseline corresponding 
                 to the coherences. Units must match wavelength (e.g., meters).
                 Shape: (N_baselines, 3)
                 Convention: Baseline b_ij = r_i - r_j corresponds to coherence V_ij.
    - wavelength: float. The radar/radio wavelength in the same units as baselines.
    - grid_size: int. The resolution of the output grid (grid_size x grid_size).

    Returns:
    - L: 2D array of x-direction cosines.
    - M: 2D array of y-direction cosines.
    - P: 2D array of beamformed power. Masked (NaN) outside the unit circle (horizon).
    """
    coherences = np.array(coherences)
    baselines = np.array(baselines)
    
    if coherences.shape[0] != len(baselines):
        raise ValueError(f"Number of coherences dim 0 ({coherences.shape[0]}) must match number of baselines ({len(baselines)})")

    # Check for batch mode
    is_batch = coherences.ndim > 1
    if is_batch:
        # Flatten batch dimensions: (N_baselines, N_samples)
        n_baselines, n_samples = coherences.shape
    else:
        n_baselines = len(coherences)
        n_samples = 1
        coherences = coherences.reshape(n_baselines, 1)

    # 1. Create grid of direction cosines (l, m)
    l = np.linspace(-1, 1, grid_size)
    m = np.linspace(-1, 1, grid_size)
    L, M = np.meshgrid(l, m)
    
    # 2. Mask for the sky
    R2 = L**2 + M**2
    mask = R2 <= 1.0
    
    # Grid points in valid region
    L_valid = L[mask]
    M_valid = M[mask]
    N_valid = np.sqrt(1 - L_valid**2 - M_valid**2)
    
    n_valid_pixels = len(L_valid)
    
    # 3. Pre-compute Phase delays: (N_baselines, N_pixels)
    # k * (b . s)
    k = 2 * np.pi / wavelength
    
    # baselines: (N_bl, 3)
    # s: (3, N_pix)
    s_vectors = np.vstack((L_valid, M_valid, N_valid))
    phase_delays = k * (baselines @ s_vectors) # (N_bl, N_pix)
    
    # Steering vectors conjugates: exp(-j * phase)
    steering_vecs = np.exp(-1j * phase_delays) # (N_bl, N_pix)
    
    print(f"Beamforming with {n_baselines} baselines over {grid_size}x{grid_size} grid ({n_valid_pixels} valid pixels), batch size {n_samples}...")
    
    # 4. Compute Power
    # P = Sum_ij V_ij * exp(-j * phi_ij)
    # V: (N_bl, N_samples)
    # S: (N_bl, N_pix)
    # Result: (N_samples, N_pix) = V.T @ S
    
    # Sum_bl ( V[bl, samp] * S[bl, pix] )
    # We want Real part * 2
    
    # Use matrix multiplication
    # (N_samples, N_bl) @ (N_bl, N_pix) -> (N_samples, N_pix)
    beamformed_flat = np.matmul(coherences.T, steering_vecs)
    
    # Add conjugate baseline contribution: 2 * Real
    P_valid = 2 * beamformed_flat.real
    
    # 5. Reshape to grid
    if is_batch:
        P = np.full((n_samples, grid_size, grid_size), np.nan)
        P[:, mask] = P_valid
    else:
        P = np.full((grid_size, grid_size), np.nan)
        P[mask] = P_valid[0]
        
    return L, M, P

if __name__ == "__main__":
    import os
    
    # 1. Configuration
    wavelength = 10.0
    R = 25.0
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    antennas = np.vstack([[0, 0, 0], [[R*np.cos(a), R*np.sin(a), 0] for a in angles]])
    
    baselines = []
    for i in range(len(antennas)):
        for j in range(i+1, len(antennas)):
            baselines.append(antennas[i] - antennas[j])
    baselines = np.array(baselines)
    k = 2 * np.pi / wavelength

    # 2. Test Scenarios
    scenarios = [
        ("one_target", [
            {'l': 0.5, 'm': 0.3, 'amp': 0.5}
        ]),
        ("two_targets", [
            {'l': 0.5, 'm': 0.3, 'amp': 0.5},
            {'l': -0.3, 'm': 0.4, 'amp': 0.3}
        ]),
        ("three_targets", [
            {'l': 0.5, 'm': 0.3, 'amp': 0.5},
            {'l': -0.3, 'm': 0.4, 'amp': 0.3},
            {'l': 0.1, 'm': -0.6, 'amp': 0.2}
        ])
    ]

    fig_dir = "Figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print(f"Created directory: {fig_dir}")

    for name, sources in scenarios:
        print(f"\n" + "="*50)
        print(f" TESTING SCENARIO: {name} ({len(sources)} sources)")
        print("="*50)

        # Synthesize coherences and full covariance matrix R
        n_ant = len(antennas)
        R_mat = np.zeros((n_ant, n_ant), dtype=complex)
        np.fill_diagonal(R_mat, 1.0) # Normalized antenna power

        coherences = []
        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                b = antennas[i] - antennas[j]
                vis = 0j
                for src in sources:
                    n = np.sqrt(max(1e-9, 1 - src['l']**2 - src['m']**2))
                    vis += src['amp'] * np.exp(1j * k * (b[0]*src['l'] + b[1]*src['m'] + b[2]*n))
                coherences.append(vis)
                R_mat[i, j] = vis
                R_mat[j, i] = np.conj(vis)

        coherences = np.array(coherences)

        # 0. Single Target Detection (Bartlett + NLLS)
        print("\nTesting Single Target Detection (Bartlett + NLLS)...")
        l_one, m_one, a_one, succ_one = detect_one_target_bartlet_clean_nlls(coherences, baselines, wavelength)
        if succ_one:
            print(f"Single Target Refined: l={l_one:.6f}, m={m_one:.6f}, Amp={np.abs(a_one):.4f}")

        # 1. Beamforming (Bartlett)
        L, M, P = beamform_coherences(coherences, baselines, wavelength, grid_size=400)

        # 2. Capon Beamforming (High Res)
        _, _, P_capon = beamform_capon(R_mat, antennas, wavelength, grid_size=400)

        # 3. CLEAN subtraction for two targets
        (pos1, pos2, a1_clean, P_res) = beamform_clean_two_peaks(coherences, baselines, wavelength, grid_size=400)
        print(f"CLEAN Peaks: T1=({pos1[0]:.4f}, {pos1[1]:.4f}), T2=({pos2[0]:.4f}, {pos2[1]:.4f})")

        # 4. Joint NLLS refinement (using CLEAN guesses)
        res_j = estimate_direction_cosines_nlls_two_targets(coherences, baselines, wavelength, pos1, pos2)
        (p1, p2, a1, a2, succ) = res_j
        
        if succ:
            print(f"Joint NLLS (from CLEAN):")
            print(f"  T1: l={p1[0]:.6f}, m={p1[1]:.6f}, Amp={np.abs(a1):.4f}")
            print(f"  T2: l={p2[0]:.6f}, m={p2[1]:.6f}, Amp={np.abs(a2):.4f}")

        # 5. Joint NLLS refinement (using Capon guesses - Masking)
        print("\nTesting Capon (Masking) + Joint NLLS detected targets...")
        p1_m, p2_m, a1_m, a2_m, succ_m = detect_two_targets_capon_nlls(R_mat, coherences, antennas, wavelength)
        if succ_m:
            print(f"Joint NLLS (from Capon Masking):")
            print(f"  T1: l={p1_m[0]:.6f}, m={p1_m[1]:.6f}, Amp={np.abs(a1_m):.4f}")
            print(f"  T2: l={p2_m[0]:.6f}, m={p2_m[1]:.6f}, Amp={np.abs(a2_m):.4f}")

        # 6. Joint NLLS refinement (using Capon guesses - CLEAN)
        print("\nTesting Capon (CLEAN) + Joint NLLS detected targets...")
        p1_cc, p2_cc, a1_cc, a2_cc, succ_cc = detect_two_targets_capon_clean_nlls(coherences, antennas, wavelength)
        if succ_cc:
            print(f"Joint NLLS (from Capon CLEAN):")
            print(f"  T1: l={p1_cc[0]:.6f}, m={p1_cc[1]:.6f}, Amp={np.abs(a1_cc):.4f}")
            print(f"  T2: l={p2_cc[0]:.6f}, m={p2_cc[1]:.6f}, Amp={np.abs(a2_cc):.4f}")

        # 7. Plotting - CLEAN guess result (Bartlett CLEAN)
        plt.figure(figsize=(10, 8))
        pc = plt.pcolormesh(L, M, P, shading='auto', cmap='inferno')
        plt.colorbar(pc, label='Beamformed Power')
        plt.scatter([s['l'] for s in sources], [s['m'] for s in sources], 
                    marker='x', color='white', s=100, label='True Sources', alpha=0.8)
        if succ:
            plt.scatter(p1[0], p1[1], marker='+', color='cyan', s=150, linewidth=2, label='Bartlett-CLEAN+NLLS T1')
            plt.scatter(p2[0], p2[1], marker='+', color='lime', s=150, linewidth=2, label='Bartlett-CLEAN+NLLS T2')
        theta = np.linspace(0, 2*np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=1)
        plt.title(f'Scenario: {name}\nJoint NLLS (Bartlett-CLEAN Start)')
        plt.xlabel('Direction Cosine L'); plt.ylabel('Direction Cosine M')
        plt.axis('equal'); plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
        fname = f"beamforming_joint_bartlett_clean_{name}.png"
        out_path = os.path.join(fig_dir, fname)
        plt.savefig(out_path, dpi=150); print(f"Saved plot: {out_path}"); plt.close()

        # 8. Plotting - Capon guess result (Capon CLEAN)
        plt.figure(figsize=(10, 8))
        pc_cc = plt.pcolormesh(L, M, P, shading='auto', cmap='inferno')
        plt.colorbar(pc_cc, label='Beamformed Power')
        plt.scatter([s['l'] for s in sources], [s['m'] for s in sources], 
                    marker='x', color='white', s=100, label='True Sources', alpha=0.8)
        if succ_cc:
            plt.scatter(p1_cc[0], p1_cc[1], marker='+', color='cyan', s=150, linewidth=2, label='Capon-CLEAN+NLLS T1')
            plt.scatter(p2_cc[0], p2_cc[1], marker='+', color='lime', s=150, linewidth=2, label='Capon-CLEAN+NLLS T2')
        plt.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=1)
        plt.title(f'Scenario: {name}\nJoint NLLS (Capon-CLEAN Start)')
        plt.xlabel('Direction Cosine L'); plt.ylabel('Direction Cosine M')
        plt.axis('equal'); plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
        fname_cc = f"beamforming_joint_capon_clean_{name}.png"
        out_path_cc = os.path.join(fig_dir, fname_cc)
        plt.savefig(out_path_cc, dpi=150); print(f"Saved plot: {out_path_cc}"); plt.close()

        # 9. Plot Capon Map
        plt.figure(figsize=(10, 8))
        pc_cap = plt.pcolormesh(L, M, P_capon, shading='auto', cmap='inferno')
        plt.colorbar(pc_cap, label='Capon Power')
        plt.scatter([s['l'] for s in sources], [s['m'] for s in sources], 
                    marker='x', color='white', s=100, label='True Sources', alpha=0.8)
        plt.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=1)
        plt.title(f'Capon Beamforming (MVDR): {name}')
        plt.xlabel('Direction Cosine L')
        plt.ylabel('Direction Cosine M')
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        fname_cap = f"beamforming_capon_{name}.png"
        out_path_cap = os.path.join(fig_dir, fname_cap)
        plt.savefig(out_path_cap, dpi=150)
        print(f"Saved plot: {out_path_cap}")
        plt.close()

    print("\nAll tests completed.")

def beamform_mimo_coupled(R_mimo, tx_antennas, rx_antennas, total_range, tx_gps, rx_gps, wavelength, grid_size=200):
    """
    Coupled Geodetic MIMO beamforming over a 2D Tx (L, M) search grid.
    For each (L_tx, M_tx), the true 3D location is found from total_range.
    Then the corresponding Rx direction cosines (L_rx, M_rx, N_rx) are derived 
    to build the MIMO steering vector and the power is evaluated.
    Returns: L_tx, M_tx, P (Coupled Power grid).
    """
    import numpy as np
    from coordinates import geolocation_from_bistatic_peak, wgs84_lla_to_ecef, ecef_to_enu_rotation
    
    l_tx = np.linspace(-1, 1, grid_size)
    m_tx = np.linspace(-1, 1, grid_size)
    L_tx, M_tx = np.meshgrid(l_tx, m_tx)
    R2_tx = L_tx**2 + M_tx**2
    mask = R2_tx <= 1.0
    
    L_v = L_tx[mask]
    M_v = M_tx[mask]
    N_v = np.sqrt(1.0 - L_v**2 - M_v**2)
    s_tx = np.vstack((L_v, M_v, N_v)) # (3, N_pix)
    
    # Geolocation to find corresponding Rx directions
    lat_tgt, lon_tgt, alt_tgt = geolocation_from_bistatic_peak(L_v, M_v, total_range, tx_gps, rx_gps)
    
    valid_tgt = ~np.isnan(lat_tgt)
    n_valid = np.sum(valid_tgt)
    P_val = np.zeros(n_valid)
    
    if n_valid > 0:
        rx_pos_ecef = np.array(wgs84_lla_to_ecef(*rx_gps))
        
        tgt_x, tgt_y, tgt_z = wgs84_lla_to_ecef(lat_tgt[valid_tgt], lon_tgt[valid_tgt], alt_tgt[valid_tgt])
        tgt_ecef = np.vstack((tgt_x, tgt_y, tgt_z)).T # (N_valid, 3)
        
        vec_rx_tgt = tgt_ecef - rx_pos_ecef
        dist_rx_tgt = np.linalg.norm(vec_rx_tgt, axis=1, keepdims=True)
        dir_ecef = vec_rx_tgt / dist_rx_tgt # (N_valid, 3)
        
        R_ecef2enu_rx = ecef_to_enu_rotation(rx_gps[0], rx_gps[1]) # (3, 3)
        dir_enu = dir_ecef @ R_ecef2enu_rx.T # (N_valid, 3)
        
        L_rx = dir_enu[:, 0]
        M_rx = dir_enu[:, 1]
        N_rx = dir_enu[:, 2] 
        s_rx = np.vstack((L_rx, M_rx, N_rx)) # (3, N_valid)
        
        k = 2 * np.pi / wavelength
        phase_tx = k * (tx_antennas @ s_tx[:, valid_tgt]) # (N_tx, N_valid)
        A_tx = np.exp(1j * phase_tx) # (N_tx, N_valid)
        
        phase_rx = k * (rx_antennas @ s_rx) # (N_rx, N_valid)
        A_rx = np.exp(1j * phase_rx) # (N_rx, N_valid)
        
        N_tx = tx_antennas.shape[0]
        N_rx_ants = rx_antennas.shape[0]
        
        # PCA clean: Isolate the dominant spatial signature from R_mimo to suppress the noise subspace
        evals, evecs = np.linalg.eigh(R_mimo)
        e1 = evecs[:, -1] # (N_vir,) Dominant eigenvector
        
        A_mimo = np.zeros((N_tx * N_rx_ants, n_valid), dtype=complex)
        for r in range(N_rx_ants):
            for t in range(N_tx):
                A_mimo[r * N_tx + t, :] = A_tx[t, :] * A_rx[r, :]
                
        # Projection of the Coupled Geodetic Steering Vector onto the isolated MIMO signal subspace
        # This perfectly matches the SNR quality of the Independent solution
        P_val = evals[-1] * np.abs(np.conj(A_mimo).T @ e1)**2 # (N_valid,)
    
    P_valid_full = np.full(len(L_v), np.nan)
    P_valid_full[valid_tgt] = P_val
    
    P = np.full((grid_size, grid_size), np.nan)
    P[mask] = P_valid_full
    
    L_rx_full = np.full(len(L_v), np.nan)
    M_rx_full = np.full(len(L_v), np.nan)
    if 'L_rx' in locals() and 'M_rx' in locals() and n_valid > 0:
        L_rx_full[valid_tgt] = L_rx
        M_rx_full[valid_tgt] = M_rx
        
    L_rx_map = np.full((grid_size, grid_size), np.nan)
    M_rx_map = np.full((grid_size, grid_size), np.nan)
    L_rx_map[mask] = L_rx_full
    M_rx_map[mask] = M_rx_full
    
    return L_tx, M_tx, P, L_rx_map, M_rx_map

def beamform_mimo_independent(R_mimo, tx_antennas, rx_antennas, wavelength, grid_size=50):
    """
    Independent MIMO beamforming over a 4D grid (L_tx, M_tx, L_rx, M_rx).
    To conserve memory, it evaluates the power across the Cartesian grid and returns the 
    global maximum peak parameters directly, along with projected 2D maps.
    Returns: (L_tx_max, M_tx_max), (L_rx_max, M_rx_max), P_tx_map, P_rx_map
    """
    import numpy as np
    k = 2 * np.pi / wavelength
    l_ax = np.linspace(-1, 1, grid_size)
    m_ax = np.linspace(-1, 1, grid_size)
    L, M = np.meshgrid(l_ax, m_ax)
    mask = (L**2 + M**2) <= 1.0
    
    L_v = L[mask]
    M_v = M[mask]
    N_v = np.sqrt(1.0 - L_v**2 - M_v**2)
    s_v = np.vstack((L_v, M_v, N_v)) # (3, N_pix)
    
    A_tx = np.exp(1j * k * (tx_antennas @ s_v)) # (N_tx, N_pix)
    A_rx = np.exp(1j * k * (rx_antennas @ s_v)) # (N_rx, N_pix)
    
    N_tx = tx_antennas.shape[0]
    N_rx = rx_antennas.shape[0]
    N_vir = N_tx * N_rx
    
    # 1. Find dominant eigenvector of R_mimo
    evals, evecs = np.linalg.eigh(R_mimo)
    e1 = evecs[:, -1] # Dominant eigenvector (size N_vir)
    
    # 2. Reshape into 2D block (N_rx, N_tx) assuming correlation matrix was formatted that way
    # In correlate_mimo_hdf5.py, v_mimo shape was (N_rx, N_tx).
    V_mat = e1.reshape((N_rx, N_tx))
    
    # 3. Perform SVD to decouple the signatures
    U, S, Vh = np.linalg.svd(V_mat, full_matrices=False)
    
    # U[:, 0] is the Rx signature, Vh[0, :] is the Tx signature
    u_rx = U[:, 0]
    v_tx = Vh[0, :] # Note Vh is already transposed/conjugated in numpy SVD
    
    # 4. Form independent beamforming power maps
    # A_tx is (N_tx, N_pix), A_rx is (N_rx, N_pix)
    # P_tx(p) = | A_tx[:, p]^H v_tx |^2
    P_tx_proj = np.abs(np.conj(A_tx).T @ v_tx)**2
    P_rx_proj = np.abs(np.conj(A_rx).T @ u_rx)**2
    
    # 5. Find peaks
    idx_tx_best = np.argmax(P_tx_proj)
    idx_rx_best = np.argmax(P_rx_proj)
    
    l_tx_max = L_v[idx_tx_best]
    m_tx_max = M_v[idx_tx_best]
    l_rx_max = L_v[idx_rx_best]
    m_rx_max = M_v[idx_rx_best]
    
    P_tx_map = np.full((grid_size, grid_size), np.nan)
    P_tx_map[mask] = P_tx_proj
    
    P_rx_map = np.full((grid_size, grid_size), np.nan)
    P_rx_map[mask] = P_rx_proj
    
    return (l_tx_max, m_tx_max), (l_rx_max, m_rx_max), P_tx_map, P_rx_map

def estimate_mimo_independent_nlls(R_mimo, tx_antennas, rx_antennas, wavelength, initial_guess_tx, initial_guess_rx):
    """
    Refines independent MIMO direction cosines using Non-Linear Least Squares (NLLS).
    
    Fits for (l_tx, m_tx, l_rx, m_rx, A_real, A_imag).
    """
    from scipy.optimize import least_squares
    import numpy as np
    
    N_tx = tx_antennas.shape[0]
    N_rx = rx_antennas.shape[0]
    N_vir = N_tx * N_rx
    
    coherences = []
    bl_tx = []
    bl_rx = []
    
    # Extract unique off-diagonal elements
    for i in range(N_vir):
        # i corresponds to r1, t1 because v_mimo shape was (N_rx, N_tx)
        r1 = i // N_tx
        t1 = i % N_tx
        for j in range(i + 1, N_vir):
            r2 = j // N_tx
            t2 = j % N_tx
            coherences.append(R_mimo[i, j])
            bl_tx.append(tx_antennas[t1] - tx_antennas[t2])
            bl_rx.append(rx_antennas[r1] - rx_antennas[r2])
            
    coherences = np.array(coherences)
    bl_tx = np.array(bl_tx)
    bl_rx = np.array(bl_rx)
    
    k = 2 * np.pi / wavelength
    
    def residuals(params):
        l_tx, m_tx, l_rx, m_rx, ar, ai = params
        r2_tx = l_tx**2 + m_tx**2
        r2_rx = l_rx**2 + m_rx**2
        
        penalty = 0.0
        if r2_tx > 1.0: penalty += (r2_tx - 1.0)
        if r2_rx > 1.0: penalty += (r2_rx - 1.0)
        if penalty > 0:
            return np.ones(2 * len(coherences)) * 1e6 * penalty
            
        n_tx = np.sqrt(1.0 - r2_tx)
        n_rx = np.sqrt(1.0 - r2_rx)
        
        complex_amp = ar + 1j * ai
        
        phase_tx = k * (bl_tx[:, 0] * l_tx + bl_tx[:, 1] * m_tx + bl_tx[:, 2] * n_tx)
        phase_rx = k * (bl_rx[:, 0] * l_rx + bl_rx[:, 1] * m_rx + bl_rx[:, 2] * n_rx)
        
        # Model is exp(+1j * phase) to match the conjugation logic identified earlier
        model = complex_amp * np.exp(1j * (phase_tx + phase_rx))
        
        res = coherences - model
        return np.concatenate([res.real, res.imag])

    a_init = np.mean(np.abs(coherences))
    x0 = [initial_guess_tx[0], initial_guess_tx[1], initial_guess_rx[0], initial_guess_rx[1], a_init, 0.0]
    
    lower = [-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf]
    upper = [1.0, 1.0, 1.0, 1.0, np.inf, np.inf]
    
    try:
        res = least_squares(residuals, x0, bounds=(lower, upper))
        p = res.x
        return (p[0], p[1]), (p[2], p[3]), p[4] + 1j * p[5], res.success
    except Exception as e:
        print(f"MIMO NLLS Error: {e}")
        return initial_guess_tx, initial_guess_rx, 0j, False


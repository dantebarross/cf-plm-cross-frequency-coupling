import numpy as np
from scipy.signal import hilbert
import numpy.fft as fft

def compute_cf_plm(x, y, fs, B=1.0):
    """
    Computes the CF-PLM using a direct FFT-based approach.
    Returns a two-sided frequency axis (centered at zero) and the CF-PLM value.
    """
    # 1) Build interferometric signal: z(t) = exp[i*(phi_x(t) - phi_y(t))]
    x_an = hilbert(x)
    y_an = hilbert(y)
    z = (x_an * np.conjugate(y_an)) / (np.abs(x_an) * np.abs(y_an))
    z = np.nan_to_num(z)
    
    # 2) Compute FFT and two-sided PSD (correct scaling: divide by (N * fs))
    N = len(z)
    Z = fft.fft(z)
    Pxx = np.abs(Z)**2 / (N * fs)
    freqs = fft.fftfreq(N, d=1/fs)
    
    # 3) Shift so zero frequency is centered
    Pxx_shifted = fft.fftshift(Pxx)
    freqs_shifted = fft.fftshift(freqs)
    
    # 4) Find the global maximum in PSD and its corresponding frequency
    peak_idx = np.argmax(Pxx_shifted)
    f_peak = freqs_shifted[peak_idx]
    
    # 5) Integrate the PSD in [f_peak - B, f_peak + B] and over all frequencies
    mask = (freqs_shifted >= f_peak - B) & (freqs_shifted <= f_peak + B)
    power_band = np.trapz(Pxx_shifted[mask], freqs_shifted[mask])
    total_power = np.trapz(Pxx_shifted, freqs_shifted)
    
    cf_plm_val = power_band / total_power
    return cf_plm_val, freqs_shifted, Pxx_shifted, f_peak

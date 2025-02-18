import numpy as np
from scipy.signal import hilbert
import numpy.fft as fft

def compute_cf_plm(x, y, fs, B=1.0):
    """
    Computes the CF-PLM using a direct FFT-based approach
    and returns a two-sided frequency axis, centered at zero.
    """
    # 1) Build interferometric signal z(t) = e^{i [phi_x(t) - phi_y(t)]}
    x_an = hilbert(x)
    y_an = hilbert(y)
    z = (x_an * np.conjugate(y_an)) / (np.abs(x_an) * np.abs(y_an))
    z = np.nan_to_num(z)

    # 2) Compute the FFT and the two-sided PSD
    N = len(z)
    Z = fft.fft(z)               # Complex FFT
    Pxx = np.abs(Z)**2 / N       # Simple PSD = |Z|^2 / N
    freqs = fft.fftfreq(N, d=1/fs)

    # 3) Shift so that zero frequency is in the middle (i.e., negative freqs on left)
    Pxx_shifted = fft.fftshift(Pxx)
    freqs_shifted = fft.fftshift(freqs)

    # 4) Identify the global maximum in the two-sided PSD
    peak_idx = np.argmax(Pxx_shifted)
    f_peak = freqs_shifted[peak_idx]

    # 5) Integrate in [f_peak - B, f_peak + B] to get CF-PLM
    mask = (freqs_shifted >= f_peak - B) & (freqs_shifted <= f_peak + B)
    power_band = np.trapz(Pxx_shifted[mask], freqs_shifted[mask])
    total_power = np.trapz(Pxx_shifted, freqs_shifted)

    cf_plm_val = power_band / total_power

    return cf_plm_val, freqs_shifted, Pxx_shifted, f_peak

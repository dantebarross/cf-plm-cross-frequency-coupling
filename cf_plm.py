import numpy as np
import scipy.signal as signal

def compute_cf_plm(x, y, fs, B=1.0):
    x_an = signal.hilbert(x)
    y_an = signal.hilbert(y)
    z = (x_an * np.conjugate(y_an)) / (np.abs(x_an) * np.abs(y_an))
    z = np.nan_to_num(z)
    f, Pxx = signal.periodogram(z, fs, window='boxcar', scaling='density')
    peak_idx = np.argmax(Pxx)
    f_peak = f[peak_idx]
    band_idx = np.where((f >= f_peak - B) & (f <= f_peak + B))
    power_band = np.trapz(Pxx[band_idx], f[band_idx])
    total_power = np.trapz(Pxx, f)
    cf_plm_val = power_band / total_power
    return cf_plm_val, f, Pxx, f_peak

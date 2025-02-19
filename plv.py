import numpy as np
from scipy.signal import hilbert

def compute_plv(x, y):
    phase_diff = np.angle(hilbert(x)) - np.angle(hilbert(y))
    return np.abs(np.mean(np.exp(1j * phase_diff)))
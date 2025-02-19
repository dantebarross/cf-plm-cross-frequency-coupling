import numpy as np
from scipy.signal import hilbert

def compute_plv(x, y):
    phase_diff = np.angle(hilbert(x)) - np.angle(hilbert(y))
    return np.abs(np.mean(np.exp(1j * phase_diff)))


if __name__ == "__main__":
    # Example signals
    t = np.linspace(0, 10, 1000)
    signal1 = np.exp(1j * (2 * np.pi * 1 * t + np.random.rand(len(t))))
    signal2 = np.exp(1j * (2 * np.pi * 1.5 * t + np.random.rand(len(t))))

    plv_value = compute_plv(signal1, signal2)
    print("PLV value:", plv_value)

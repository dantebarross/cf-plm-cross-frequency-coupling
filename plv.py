import numpy as np

def compute_plv(signal1, signal2):
    phase1 = np.angle(signal1)
    phase2 = np.angle(signal2)
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

if __name__ == "__main__":
    # Example signals
    t = np.linspace(0, 10, 1000)
    signal1 = np.exp(1j * (2 * np.pi * 1 * t + np.random.rand(len(t))))
    signal2 = np.exp(1j * (2 * np.pi * 1.5 * t + np.random.rand(len(t))))

    plv_value = compute_plv(signal1, signal2)
    print("PLV value:", plv_value)

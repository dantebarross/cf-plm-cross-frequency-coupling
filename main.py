import numpy as np
import matplotlib.pyplot as plt
from cf_plm import compute_cf_plm
from rossler import simulate_rossler
from plv import compute_plv
from kuramoto import kuramoto_simulation

# Parameters for Rössler simulation
f1 = 1.5  # Frequency within 0.5-2 Hz
f2 = 45  # Frequency within 30-60 Hz
c = 0.02
T = 100  # Reduced simulation time
fs = 250  # Reduced sampling frequency
init_state = [1, 1, 1, 1, 1, 1]

# Simulate Rössler attractors
rossler_data, t_eval = simulate_rossler(f1, f2, c, T, fs, init_state)
x1 = rossler_data[0, :]
y2 = rossler_data[4, :]
Delta_f = 7
y2_shifted = y2 * np.exp(-1j * 2*np.pi * Delta_f * t_eval)

# Compute CF-PLM for Rössler
cf_plm_value_rossler, freqs_rossler, Pxx_rossler, peak_rossler = compute_cf_plm(x1, np.real(y2_shifted), fs)
print("CF-PLM value (Rössler):", cf_plm_value_rossler)

# Parameters for Kuramoto simulation with different frequencies
T_kuramoto = 100
dt = 0.001
freqs = np.array([1.5, 45]) + np.random.normal(0, 0.5, 2)  # Add variability to frequencies
tau = 0.6

# Simulate Kuramoto oscillators with coupling
k_with_coupling = 1.0
theta_with_coupling = kuramoto_simulation(T_kuramoto, dt, freqs, k_with_coupling, tau)
signals_with_coupling = np.sin(theta_with_coupling)

# Compute CF-PLM for Kuramoto with coupling
cf_plm_value_kuramoto_with_coupling, freqs_kuramoto_with_coupling, Pxx_kuramoto_with_coupling, peak_kuramoto_with_coupling = compute_cf_plm(signals_with_coupling[0, :], signals_with_coupling[1, :], 1/dt)
print("CF-PLM value (Kuramoto with coupling):", cf_plm_value_kuramoto_with_coupling)

# Compute PLV for Kuramoto with coupling
plv_value_kuramoto_with_coupling = compute_plv(signals_with_coupling[0, :], signals_with_coupling[1, :])
print("PLV value (Kuramoto with coupling):", plv_value_kuramoto_with_coupling)

# Parameters for Kuramoto simulation with same frequencies
freqs_same = np.array([10, 10])  # Same frequencies
theta_same_freq = kuramoto_simulation(T_kuramoto, dt, freqs_same, k_with_coupling, tau)
signals_same_freq = np.sin(theta_same_freq)

# Compute CF-PLM for Kuramoto with same frequencies
cf_plm_value_kuramoto_same_freq, freqs_kuramoto_same_freq, Pxx_kuramoto_same_freq, peak_kuramoto_same_freq = compute_cf_plm(signals_same_freq[0, :], signals_same_freq[1, :], 1/dt)
print("CF-PLM value (Kuramoto with same frequencies):", cf_plm_value_kuramoto_same_freq)

# Compute PLV for Kuramoto with same frequencies
plv_value_kuramoto_same_freq = compute_plv(signals_same_freq[0, :], signals_same_freq[1, :])
print("PLV value (Kuramoto with same frequencies):", plv_value_kuramoto_same_freq)

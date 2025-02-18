import numpy as np
import matplotlib.pyplot as plt
from cf_plm import compute_cf_plm
from rossler import simulate_rossler
from kuramoto import kuramoto_simulation

# Parameters for Rössler simulation
f1 = 1.5  # Frequency within 0.5-2 Hz
f2 = 45  # Frequency within 30-60 Hz
c = 0.02
T = 420
fs = 625
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

# Parameters for Kuramoto simulation
T_kuramoto = 100
dt = 0.001
freqs = np.array([1.5, 45])  # Frequencies within 0.5-2 Hz and 30-60 Hz
k = 1.0
tau = 0.6

# Simulate Kuramoto oscillators
theta = kuramoto_simulation(T_kuramoto, dt, freqs, k, tau)
time_k = np.arange(0, T_kuramoto, dt)
signals = np.sin(theta)

# Compute CF-PLM for Kuramoto
cf_plm_value_kuramoto, freqs_kuramoto, Pxx_kuramoto, peak_kuramoto = compute_cf_plm(signals[0, :], signals[1, :], 1/dt)
print("CF-PLM value (Kuramoto):", cf_plm_value_kuramoto)

# # Plot the generated signals (zoom in the initial 5 seconds)
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t_eval[:5*fs], x1[:5*fs], label='x1 (Rössler)')
# plt.plot(t_eval[:5*fs], np.real(y2_shifted[:5*fs]), label='y2_shifted (Rössler)')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Rössler Attractors (Initial 5 seconds)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(time_k[:5*int(1/dt)], signals[0, :5*int(1/dt)], label='Oscillator 1 (Kuramoto)')
# plt.plot(time_k[:5*int(1/dt)], signals[2, :5*int(1/dt)], label='Oscillator 3 (Kuramoto)')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Kuramoto Oscillators (Initial 5 seconds)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Plot the power spectral density (PSD) of the interferometric signal
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(freqs_rossler, Pxx_rossler)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Power Spectral Density')
# plt.title('Interferometric PSD (Rössler)')

# plt.subplot(2, 1, 2)
# plt.plot(freqs_kuramoto, Pxx_kuramoto)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Power Spectral Density')
# plt.title('Interferometric PSD (Kuramoto)')

# plt.tight_layout()
# plt.show()

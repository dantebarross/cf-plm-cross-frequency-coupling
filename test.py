import numpy as np
import matplotlib.pyplot as plt
from cf_plm import compute_cf_plm
from rossler import simulate_rossler
from kuramoto import kuramoto_simulation

def test_rossler_multiple_runs(c_values, runs=3):
    """Test Rössler system with multiple coupling values and average over runs."""
    avg_results = np.zeros(len(c_values))
    for r in range(runs):
        results = []
        for c in c_values:
            rossler_data, _ = simulate_rossler(f1=10, f2=10, c=c, T=200, fs=500,
                                               init_state=[1, 1, 1, 1.1, 1, 1])
            x1 = rossler_data[0, :]
            y2 = rossler_data[4, :]
            cf_plm_val, _, _, _ = compute_cf_plm(x1, y2, 500)
            results.append(cf_plm_val)
        avg_results += np.array(results)
    return avg_results / runs

def test_kuramoto_multiple_runs(k_values, runs=3, N=100, noise=0.01):
    """Test Kuramoto system with multiple coupling values and average over runs."""
    avg_results = np.zeros(len(k_values))
    for r in range(runs):
        results = []
        for k in k_values:
            freqs = np.random.normal(loc=10, scale=noise, size=N)  # Add variability
            theta = kuramoto_simulation(T=100, dt=0.001, freqs=freqs, k=k, tau=0.5)
            signals = np.sin(theta)
            cf_plm_val, _, _, _ = compute_cf_plm(signals[0, :], signals[-1, :], 1000)
            results.append(cf_plm_val)
        avg_results += np.array(results)
    return avg_results / runs

# ---------------------------
# Test Rössler
c_vals = np.linspace(0, 0.05, 20)
rossler_results = test_rossler_multiple_runs(c_vals, runs=5)

plt.figure(figsize=(8, 4))
plt.plot(c_vals, rossler_results, '-o', label='Rössler')
plt.xlabel("Coupling c")
plt.ylabel("CF-PLM")
plt.title("Rössler CF-PLM vs. Coupling c (Average of 5 runs)")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# Test Kuramoto
k_vals = np.linspace(0, 2, 20)
kuramoto_results = test_kuramoto_multiple_runs(k_vals, runs=5, N=100, noise=0.05)

plt.figure(figsize=(8, 4))
plt.plot(k_vals, kuramoto_results, '-o', color='orange', label='Kuramoto')
plt.xlabel("Coupling k")
plt.ylabel("CF-PLM")
plt.title("Kuramoto CF-PLM vs. Coupling k (Average of 5 runs)")
plt.grid(True)
plt.legend()
plt.show()

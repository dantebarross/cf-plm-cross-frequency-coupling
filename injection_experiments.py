import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from cf_plm import compute_cf_plm
from plv import compute_plv
from kuramoto import kuramoto_simulation
from scipy.interpolate import interp1d

# -----------------------------
# Load ECG data (already in 0.5-2 Hz band) from CSV
ecg_path = "athlete_1_post_cathodic_processed_ecg.csv"
ecg_df = pd.read_csv(ecg_path, header=0)
if "time" in ecg_df.columns:
    time_ecg = ecg_df["time"].astype(float).values.flatten()
    fs_ecg = 1 / np.mean(np.diff(time_ecg))
else:
    raise ValueError("Time column not found in ECG data")

if "ecg" in ecg_df.columns:
    ecg = ecg_df["ecg"].astype(float).values.flatten()
else:
    ecg = ecg_df.iloc[:, 1].astype(float).values.flatten()

# -----------------------------
# Load EEG data from FIF and filter to 30-60 Hz
eeg_path = "athlete_1_post_cathodic.fif"
raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose="ERROR")
raw.filter(30, 60, fir_design="firwin", verbose="ERROR")
eeg = raw.get_data(picks=[0]).flatten()

# -----------------------------
# Define injection parameters
# We'll use a fixed high injection amplitude for all cases.
amp = 0.002  
# Injection signal frequencies for ECG and EEG
ecg_inj_freq = 1    # Hz for ECG
eeg_inj_freq = 40   # Hz for EEG

fs_ecg = 1 / np.mean(np.diff(time_ecg))  # Automatically detect sampling frequency for ECG
fs_eeg = int(raw.info["sfreq"])
duration = min(len(ecg) / fs_ecg, len(eeg) / fs_eeg)  # in seconds
t_ecg = np.linspace(0, duration, int(duration * fs_ecg), endpoint=False)
t_eeg = np.linspace(0, duration, int(duration * fs_eeg), endpoint=False)

ecg = ecg[: int(duration * fs_ecg)]
eeg = eeg[: int(duration * fs_eeg)]

# Define injection percentages including 0%
injection_percentages = {
    "0%": 0.0,
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.0
}

# Fixed pulse duration for injection segments in seconds
pulse_duration = 1.0  
pulse_samples = int(pulse_duration * fs_ecg)

# Prepare dictionary to hold CF-PLM and PLV results for each injection level
cf_plm_results = {}
plv_results = {}

# Loop over injection percentages
for label, pct in injection_percentages.items():
    # Determine total number of samples to inject
    total_inj_samples = int(pct * len(t_ecg))
    # Determine number of pulses; ensure at least one pulse
    num_pulses = max(1, total_inj_samples // pulse_samples)
    spacing = len(t_ecg) // num_pulses

    # Create injection mask for ECG and EEG (same mask used for phase synchronization)
    inj_mask = np.zeros_like(t_ecg)
    injection_intervals = []  # To record intervals for highlighting
    for i in range(num_pulses):
        start_idx = i * spacing
        end_idx = start_idx + pulse_samples
        if end_idx > len(inj_mask):
            end_idx = len(inj_mask)
        inj_mask[start_idx:end_idx] = 1
        injection_intervals.append((t_ecg[start_idx], t_ecg[end_idx - 1]))

    # Create continuous injection sinusoids for ECG and EEG (using respective time vectors)
    inj_ecg_full = amp * np.sin(2 * np.pi * ecg_inj_freq * t_ecg)
    inj_eeg_full = amp * np.sin(2 * np.pi * eeg_inj_freq * t_eeg)
    
    # Create injection mask for EEG based on injection intervals (using t_eeg)
    inj_mask_eeg = np.zeros_like(t_eeg)
    for start, end in injection_intervals:
        inj_mask_eeg[(t_eeg >= start) & (t_eeg < end)] = 1
    
    # Apply the injection masks
    inj_ecg = inj_ecg_full * inj_mask
    inj_eeg = inj_eeg_full * inj_mask_eeg
    
    # Inject into original signals
    ecg_injected = ecg + inj_ecg
    eeg_injected = eeg + inj_eeg
    
    # Resample EEG signal to match ECG sampling rate using linear interpolation
    interp_func = interp1d(t_eeg, eeg_injected, kind="linear", fill_value="extrapolate")
    eeg_resampled = interp_func(t_ecg)
    
    # Compute CF-PLM and PLV metrics
    cf_plm_val, freqs, Pxx, f_peak = compute_cf_plm(ecg_injected, eeg_resampled, fs_ecg)
    cf_plm_results[label] = cf_plm_val
    print(f"CF-PLM ({label} Injection):", cf_plm_val)
    
    plv_val = compute_plv(ecg_injected, eeg_resampled)
    plv_results[label] = plv_val
    print(f"PLV ({label} Injection):", plv_val)

    # Debug information for low injection percentages
# Summary: Print final CF-PLM and PLV metrics for each injection percentage
print("\nSummary of CF-PLM and PLV Metrics vs Injection Percentage:")
for level in cf_plm_results.keys():
    print(f"{level}: CF-PLM: {cf_plm_results[level]}, PLV: {plv_results[level]}")

# Additional scenario with same frequencies
T_kuramoto = 100
dt = 0.001
k_with_coupling = 1.0
tau = 0.6

# Prepare dictionary to hold CF-PLM and PLV results for each injection level
cf_plm_results_same_freq = {}
plv_results_same_freq = {}

# Loop over injection percentages for same frequencies
for label, pct in injection_percentages.items():
    # Determine total number of samples to inject
    total_inj_samples = int(pct * len(t_ecg))
    # Determine number of pulses; ensure at least one pulse
    num_pulses = max(1, total_inj_samples // pulse_samples)
    spacing = len(t_ecg) // num_pulses

    # Create injection mask for ECG and EEG (same mask used for phase synchronization)
    inj_mask = np.zeros_like(t_ecg)
    injection_intervals = []  # To record intervals for highlighting
    for i in range(num_pulses):
        start_idx = i * spacing
        end_idx = start_idx + pulse_samples
        if end_idx > len(inj_mask):
            end_idx = len(inj_mask)
        inj_mask[start_idx:end_idx] = 1
        injection_intervals.append((t_ecg[start_idx], t_ecg[end_idx - 1]))

    # Create continuous injection sinusoids for ECG and EEG (using respective time vectors)
    inj_ecg_full = amp * np.sin(2 * np.pi * 10 * t_ecg)  # Use same frequency for ECG
    inj_eeg_full = amp * np.sin(2 * np.pi * 10 * t_eeg)  # Use same frequency for EEG
    
    # Create injection mask for EEG based on injection intervals (using t_eeg)
    inj_mask_eeg = np.zeros_like(t_eeg)
    for start, end in injection_intervals:
        inj_mask_eeg[(t_eeg >= start) & (t_eeg < end)] = 1
    
    # Apply the injection masks
    inj_ecg = inj_ecg_full * inj_mask
    inj_eeg = inj_eeg_full * inj_mask_eeg
    
    # Inject into original signals
    ecg_injected = ecg + inj_ecg
    eeg_injected = eeg + inj_eeg
    
    # Resample EEG signal to match ECG sampling rate using linear interpolation
    interp_func = interp1d(t_eeg, eeg_injected, kind="linear", fill_value="extrapolate")
    eeg_resampled = interp_func(t_ecg)
    
    # Compute CF-PLM and PLV metrics
    cf_plm_val, freqs, Pxx, f_peak = compute_cf_plm(ecg_injected, eeg_resampled, fs_ecg)
    cf_plm_results_same_freq[label] = cf_plm_val
    print(f"CF-PLM ({label} Injection, Same Frequencies):", cf_plm_val)
    
    plv_val = compute_plv(ecg_injected, eeg_resampled)
    plv_results_same_freq[label] = plv_val
    print(f"PLV ({label} Injection, Same Frequencies):", plv_val)

# Summary: Print final CF-PLM and PLV metrics for each injection percentage for same frequencies
print("\nSummary of CF-PLM and PLV Metrics vs Injection Percentage (Same Frequencies):")
for level in cf_plm_results_same_freq.keys():
    print(f"{level}: CF-PLM: {cf_plm_results_same_freq[level]}, PLV: {plv_results_same_freq[level]}")

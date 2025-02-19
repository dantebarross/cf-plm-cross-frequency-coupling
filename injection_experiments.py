import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from cf_plm import compute_cf_plm
from plv import compute_plv
from kuramoto import kuramoto_simulation
from injection_utils import (
    generate_injection_mask,
    generate_injection_signal,
    apply_injection,
    resample_signal
)

# -----------------------------
# Load ECG data (already filtered in 0.5-2 Hz band) from CSV
ecg_path = "subject_1_ecg_data.csv"
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
# Load EEG data from FIF file and filter to 30-60 Hz
eeg_path = "subject_1_eeg_data.fif"
raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose="ERROR")
raw.filter(30, 60, fir_design="firwin", verbose="ERROR")
eeg = raw.get_data(picks=[0]).flatten()

# -----------------------------
# Define common time vectors for ECG and EEG
duration = min(len(ecg) / fs_ecg, len(eeg) / raw.info["sfreq"])  # in seconds
t_ecg = np.linspace(0, duration, int(duration * fs_ecg), endpoint=False)
t_eeg = np.linspace(0, duration, int(duration * raw.info["sfreq"]), endpoint=False)

ecg = ecg[: int(duration * fs_ecg)]
eeg = eeg[: int(duration * raw.info["sfreq"])]

# -----------------------------
# Compute baseline CF-PLM between the raw ECG and EEG
# Resample EEG to ECG's sampling rate
eeg_resampled_baseline = resample_signal(t_eeg, eeg, t_ecg)
baseline_cf_plm, _, _, _ = compute_cf_plm(ecg, eeg_resampled_baseline, fs_ecg)
print(f"Baseline CF-PLM (ECG vs EEG, no injection): {baseline_cf_plm}")

# -----------------------------
# Define injection parameters
# Instead of using a fixed amp, we now scale amplitude relative to the signal's std.
# Here we choose a scale factor (e.g., 5) to make the injected oscillations dominant.
scale_factor_ecg = 5 * np.std(ecg)
scale_factor_eeg = 5 * np.std(eeg)

# For Experiment 1: different injection frequencies (ECG: 1Hz, EEG: 40Hz)
ecg_inj_freq = 1.0     # Hz for ECG injection
eeg_inj_freq = 40.0    # Hz for EEG injection

# Injection percentages (0% to 100%)
injection_percentages = {
    "0%": 0.0,
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.0
}

pulse_duration = 1.0  # seconds for each injection burst

# Dictionaries to store results
cf_plm_results = {}
plv_results = {}

# -----------------------------
# Experiment 1: Injection with different frequencies for ECG and EEG
for label, pct in injection_percentages.items():
    # Generate injection mask for ECG based on its time vector
    inj_mask, injection_intervals = generate_injection_mask(t_ecg, pulse_duration, pct, fs_ecg)
    
    # Scale amplitude for ECG injection based on standard deviation and injection pct
    amp_ecg = scale_factor_ecg * np.sqrt(pct)
    inj_ecg_full = generate_injection_signal(t_ecg, ecg_inj_freq, amp_ecg, phase=0.0)
    
    # Apply injection to ECG
    ecg_injected = apply_injection(ecg, inj_ecg_full, inj_mask)
    
    # For EEG, use the same injection intervals to synchronize phases
    inj_mask_eeg = np.zeros_like(t_eeg)
    for start, end in injection_intervals:
        inj_mask_eeg[(t_eeg >= start) & (t_eeg < end)] = 1
    
    amp_eeg = scale_factor_eeg * np.sqrt(pct)
    inj_eeg_full = generate_injection_signal(t_eeg, eeg_inj_freq, amp_eeg, phase=0.0)
    
    # Apply injection to EEG
    eeg_injected = apply_injection(eeg, inj_eeg_full, inj_mask_eeg)
    
    # Resample EEG to match ECG sampling rate for CF-PLM calculation
    eeg_injected_resampled = resample_signal(t_eeg, eeg_injected, t_ecg)
    
    # Compute CF-PLM and PLV between the injected signals
    cf_plm_val, _, _, _ = compute_cf_plm(ecg_injected, eeg_injected_resampled, fs_ecg)
    plv_val = compute_plv(ecg_injected, eeg_injected_resampled)
    
    cf_plm_results[label] = cf_plm_val
    plv_results[label] = plv_val

# Summary for Experiment 1
print("\nExperiment 1: ECG freq=1Hz, EEG freq=40Hz")
for level in cf_plm_results.keys():
    print(f"{level} Injection => CF-PLM: {cf_plm_results[level]}, PLV: {plv_results[level]}")

# -----------------------------
# Experiment 2: Injection with same frequencies for both signals (10 Hz)
cf_plm_results_same_freq = {}
plv_results_same_freq = {}

for label, pct in injection_percentages.items():
    # Generate injection mask for ECG
    inj_mask, injection_intervals = generate_injection_mask(t_ecg, pulse_duration, pct, fs_ecg)
    
    amp_ecg = scale_factor_ecg * np.sqrt(pct)
    inj_ecg_full = generate_injection_signal(t_ecg, 10.0, amp_ecg, phase=0.0)
    ecg_injected = apply_injection(ecg, inj_ecg_full, inj_mask)
    
    # Generate matching injection mask for EEG
    inj_mask_eeg = np.zeros_like(t_eeg)
    for start, end in injection_intervals:
        inj_mask_eeg[(t_eeg >= start) & (t_eeg < end)] = 1
    
    amp_eeg = scale_factor_eeg * np.sqrt(pct)
    inj_eeg_full = generate_injection_signal(t_eeg, 10.0, amp_eeg, phase=0.0)
    eeg_injected = apply_injection(eeg, inj_eeg_full, inj_mask_eeg)
    
    # Resample EEG to match ECG time base
    eeg_injected_resampled = resample_signal(t_eeg, eeg_injected, t_ecg)
    
    # Compute CF-PLM and PLV
    cf_plm_val, _, _, _ = compute_cf_plm(ecg_injected, eeg_injected_resampled, fs_ecg)
    plv_val = compute_plv(ecg_injected, eeg_injected_resampled)
    
    cf_plm_results_same_freq[label] = cf_plm_val
    plv_results_same_freq[label] = plv_val

# Summary for Experiment 2
print("\nExperiment 2: Both signals injected at 10 Hz")
for level in cf_plm_results_same_freq.keys():
    print(f"{level} Injection => CF-PLM: {cf_plm_results_same_freq[level]}, PLV: {plv_results_same_freq[level]}")

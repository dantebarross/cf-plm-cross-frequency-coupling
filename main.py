import numpy as np
from cf_plm import compute_cf_plm
from plv import compute_plv
from injection_utils import (
    generate_injection_mask,
    generate_injection_signal,
    apply_injection
)
from scipy.signal import butter, sosfiltfilt

def generate_pink_noise(N, seed=None):
    """
    Generate pink noise (1/f noise) by scaling the FFT of white noise.
    """
    if seed is not None:
        np.random.seed(seed)
    white = np.random.randn(N)
    X_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N)
    # Avoid division by zero at f=0
    freqs[0] = 1.0
    X_pink = X_white / np.sqrt(freqs)
    pink = np.fft.irfft(X_pink, n=N)
    pink -= np.mean(pink)
    pink /= np.std(pink)
    return pink

def simulate_eeg_like_signal(T, fs, seed=None):
    """
    Generate an EEG-like signal by bandpass filtering pink noise in 0.5–60 Hz.
    A 2nd-order Butterworth filter is used to preserve phase variability.
    """
    N = int(T * fs)
    raw_signal = generate_pink_noise(N, seed=seed)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 60.0 / nyq
    sos = butter(2, [low, high], btype='band', output='sos')
    eeg_like = sosfiltfilt(sos, raw_signal)
    eeg_like -= np.mean(eeg_like)
    eeg_like /= np.std(eeg_like)
    return eeg_like

def simulate_and_inject(injection_frequencies, injection_percentages, fs, base_signal_1, base_signal_2):
    """
    Generate sinusoidal injections on two independent base signals.
    The injection amplitude scales as std * sqrt(pct), and both injections have the same phase.
    CF-PLM and PLV are computed between the two injected signals.
    """
    results = {}
    duration = len(base_signal_1) / fs
    t_base = np.linspace(0, duration, len(base_signal_1), endpoint=False)
    pulse_duration = 1.0  # seconds
    
    for label, pct in injection_percentages.items():
        # Adaptive injection amplitude: sqrt(pct) * std
        amp1 = np.std(base_signal_1) * np.sqrt(pct)
        amp2 = np.std(base_signal_2) * np.sqrt(pct)
        
        # Generate injection signals with same phase (0) for strong phase locking
        inj_signal_1 = generate_injection_signal(t_base, injection_frequencies[0], amp1, phase=0.0)
        inj_signal_2 = generate_injection_signal(t_base, injection_frequencies[1], amp2, phase=0.0)
        
        # Generate injection masks (same timing for both channels)
        inj_mask_1, intervals = generate_injection_mask(t_base, pulse_duration, pct, fs)
        inj_mask_2 = np.zeros_like(t_base)
        for start, end in intervals:
            inj_mask_2[(t_base >= start) & (t_base < end)] = 1
        
        # Apply injections to the independent base signals
        injected_signal_1 = apply_injection(base_signal_1, inj_signal_1, inj_mask_1)
        injected_signal_2 = apply_injection(base_signal_2, inj_signal_2, inj_mask_2)
        
        # Compute CF-PLM and PLV between the two injected signals
        cf_plm_val, _, _, _ = compute_cf_plm(injected_signal_1, injected_signal_2, fs)
        plv_val = compute_plv(injected_signal_1, injected_signal_2)
        results[label] = (cf_plm_val, plv_val)
    
    return results

if __name__ == "__main__":
    # Simulation parameters
    T_signal = 300       # 300 seconds (5 minutes) for variability
    fs = 250             # Sampling frequency in Hz
    seed1 = 42           # Seed for base signal 1
    seed2 = 43           # Seed for base signal 2
    
    # Generate two independent EEG-like signals
    base_signal_1 = simulate_eeg_like_signal(T_signal, fs, seed=seed1)
    base_signal_2 = simulate_eeg_like_signal(T_signal, fs, seed=seed2)
    
    # Compute baseline CF-PLM between independent base signals
    baseline_cf_plm, _, _, _ = compute_cf_plm(base_signal_1, base_signal_2, fs)
    print("Baseline CF-PLM (Independent Signals):", baseline_cf_plm)
    
    # Define injection percentages
    injection_percentages = {
        "0%": 0.0,
        "25%": 0.25,
        "50%": 0.50,
        "75%": 0.75,
        "100%": 1.0
    }
    
    # Experiment 1: Different frequencies (e.g., 1 Hz for signal 1, 40 Hz for signal 2)
    freqs_exp1 = np.array([1.0, 40.0])
    results_exp1 = simulate_and_inject(freqs_exp1, injection_percentages, fs, base_signal_1, base_signal_2)
    print("\nResults for Experiment 1 (Different frequencies):")
    for level, (cf_plm_val, plv_val) in results_exp1.items():
        print(f"{level}: CF-PLM={cf_plm_val:.6f}, PLV={plv_val:.6f}")
    
    # Experiment 2: Same frequencies (both signals injected at 10 Hz)
    freqs_exp2 = np.array([10.0, 10.0])
    results_exp2 = simulate_and_inject(freqs_exp2, injection_percentages, fs, base_signal_1, base_signal_2)
    print("\nResults for Experiment 2 (Same frequencies):")
    for level, (cf_plm_val, plv_val) in results_exp2.items():
        print(f"{level}: CF-PLM={cf_plm_val:.6f}, PLV={plv_val:.6f}")

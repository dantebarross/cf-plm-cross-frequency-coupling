import numpy as np
from scipy.interpolate import interp1d

def generate_injection_mask(time_vector, pulse_duration, injection_pct, fs):
    """
    Generate an injection mask and corresponding injection intervals.
    
    [Atomic Reasoning]
    - We create a binary mask by computing the total number of samples to inject (based on the injection percentage)
      and splitting the time vector into pulses of specified duration.
    
    Parameters:
        time_vector (np.ndarray): Time axis for the signal.
        pulse_duration (float): Duration (seconds) of each injection pulse.
        injection_pct (float): Fraction (0 to 1) of total samples to be injected.
        fs (float): Sampling frequency of the signal.
        
    Returns:
        inj_mask (np.ndarray): Binary mask (same shape as time_vector) with ones where injection occurs.
        injection_intervals (list of tuple): List of (start_time, end_time) for each pulse.
    """
    total_samples = len(time_vector)
    total_inj_samples = int(injection_pct * total_samples)
    pulse_samples = int(pulse_duration * fs)
    num_pulses = max(1, total_inj_samples // pulse_samples)
    spacing = total_samples // num_pulses
    
    inj_mask = np.zeros_like(time_vector)
    injection_intervals = []
    for i in range(num_pulses):
        start_idx = i * spacing
        end_idx = start_idx + pulse_samples
        if end_idx > total_samples:
            end_idx = total_samples
        inj_mask[start_idx:end_idx] = 1
        injection_intervals.append((time_vector[start_idx], time_vector[end_idx - 1]))
    return inj_mask, injection_intervals

def generate_injection_signal(time_vector, inj_freq, amplitude, phase=0.0):
    """
    Generate a continuous sinusoidal injection signal with optional phase offset.
    
    [Atomic Reasoning]
    - A sine wave is generated at the desired frequency, amplitude, and phase.
    
    Parameters:
        time_vector (np.ndarray): Time axis over which to generate the sine.
        inj_freq (float): Frequency (Hz) of the sine wave.
        amplitude (float): Amplitude of the sine.
        phase (float): Phase offset (radians).
        
    Returns:
        inj_signal (np.ndarray): Sinusoidal injection signal.
    """
    inj_signal = amplitude * np.sin(2 * np.pi * inj_freq * time_vector + phase)
    return inj_signal

def apply_injection(original_signal, inj_signal, inj_mask):
    """
    Apply the injection signal to the original signal using the binary mask.
    
    [Atomic Reasoning]
    - Multiply the injection signal by the mask and add it to the original signal.
    
    Parameters:
        original_signal (np.ndarray): The unmodified base signal.
        inj_signal (np.ndarray): The injection sine wave.
        inj_mask (np.ndarray): Binary mask indicating when injection occurs.
        
    Returns:
        injected_signal (np.ndarray): Signal after injection.
    """
    injected_signal = original_signal + inj_signal * inj_mask
    return injected_signal

def resample_signal(source_time, signal, target_time):
    """
    Resample a signal from its original time base to a target time base using linear interpolation.
    
    [Atomic Reasoning]
    - Linear interpolation aligns signals with different sampling grids.
    
    Parameters:
        source_time (np.ndarray): Original time vector.
        signal (np.ndarray): Signal to be resampled.
        target_time (np.ndarray): New time vector.
        
    Returns:
        resampled_signal (np.ndarray): Signal interpolated to target_time.
    """
    interp_func = interp1d(source_time, signal, kind="linear", fill_value="extrapolate")
    resampled_signal = interp_func(target_time)
    return resampled_signal

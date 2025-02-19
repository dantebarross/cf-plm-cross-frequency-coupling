# CF-PLM Cross-Frequency Coupling

This repository contains code for analyzing cross-frequency phase-locking modulation (CF-PLM) in various signals. CF-PLM is a measure used to quantify the coupling between different frequency components of a signal, which is particularly useful in neuroscience for studying interactions between brain regions.

## Files

- **cf_plm.py**: Contains the implementation of the CF-PLM computation using a direct FFT-based approach.
- **main.py**: Main script that computes CF-PLM and PLV between injected signals.
- **injection_experiments.py**: Scripts for conducting injection experiments and computing CF-PLM between ECG and EEG signals.
- **test.py**: Contains test cases and plots for visualizing CF-PLM results.
- **kuramoto.py**: Kuramoto model implementation for generating synthetic signals.
- **injection_utils.py**: Utility functions for signal injection and processing.
- **rossler.py**: Rössler system implementation for generating synthetic signals.

## Usage

1. **Compute CF-PLM**: Run the `main.py` script to compute CF-PLM and PLV between injected signals.
2. **Injection Experiments**: Use the `injection_experiments.py` script to conduct injection experiments and compute CF-PLM between ECG and EEG signals.
3. **Testing and Visualization**: Use the `test.py` script to visualize CF-PLM results.

## References

For more information on CF-PLM and its applications, refer to the following article:
- [Detection of Cross-Frequency Coupling Between Brain Areas: An Extension of Phase Linearity Measurement](https://pubmed.ncbi.nlm.nih.gov/35546895/)

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

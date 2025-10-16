import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.signal import find_peaks

def read_nmr_data(file_path):
    """
    Read NMR FFT data from a .dat file
    
    Parameters:
    file_path (str): Path to the .dat file
    
    Returns:
    tuple: (frequencies, fft_real, fft_magnitude)
    """
    try:
        # Read the data, skipping the header row
        data = pd.read_csv(file_path, sep='\t', skiprows=1, 
                          names=['Freq_Hz', 'FFT_Real_V', 'FFT_Mag_V'])
        return data['Freq_Hz'].values, data['FFT_Real_V'].values, data['FFT_Mag_V'].values
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def plot_single_spectrum(freq, fft_real, fft_mag, title, save_path=None):
    """
    Plot a single NMR spectrum showing only the real component
    
    Parameters:
    freq (array): Frequency values in Hz
    fft_real (array): Real FFT component values
    fft_mag (array): Magnitude FFT component values (not used)
    title (str): Title for the plot
    save_path (str): Optional path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Find the frequency of the highest peak (maximum absolute value)
    max_peak_value = np.max(np.abs(fft_real))
    peak_idx = np.argmax(np.abs(fft_real))
    peak_freq = freq[peak_idx]
    
    # Use baseline from the end of the frequency range for more stable normalization
    # Take the last 10% of the data as baseline region
    baseline_start_idx = int(len(fft_real) * 0.9)
    baseline_region = fft_real[baseline_start_idx:]
    
    # Find the baseline level (median of the end region to avoid outliers)
    baseline_level = np.median(np.abs(baseline_region))
    
    # Find local maxima to identify actual peaks
    peaks, _ = find_peaks(np.abs(fft_real), height=max_peak_value / 10, distance=5)
    
    if len(peaks) > 0:
        # Get the peak values and find significant ones (within 1/4 of max)
        peak_values = np.abs(fft_real)[peaks]
        significant_peaks = peak_values >= max_peak_value / 4
        
        if np.any(significant_peaks):
            # Use the minimum among significant peak values
            min_significant_peak = np.min(peak_values[significant_peaks])
            # But don't go below a reasonable baseline level
            normalization_factor = max(min_significant_peak, baseline_level * 2)
        else:
            # If no significant peaks, use the baseline level as reference
            normalization_factor = max(np.min(peak_values), baseline_level * 2)
    else:
        # Fallback: use baseline level for normalization
        normalization_factor = max(baseline_level * 3, max_peak_value / 10)
    
    # Normalize the data so that the lowest significant peak has height 1
    fft_real_normalized = fft_real / normalization_factor

    # Special correction for dry data - add 0.5 to correct baseline
    if 'dry' in title.lower():
        fft_real_normalized = fft_real_normalized + 0.5
    
    # Set frequency range centered on peak, spanning 1000 Hz
    freq_span = 1000
    freq_min = peak_freq - freq_span / 2
    freq_max = peak_freq + freq_span / 2
    
    # Ensure we don't go beyond the data limits
    freq_min = max(freq_min, freq.min())
    freq_max = min(freq_max, freq.max())
    
    # Plot normalized FFT Real component
    ax.plot(freq, fft_real_normalized, 'b-', linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title(f'{title} - NMR Spectrum (Normalized)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freq_min, freq_max)
    
    # Set y-axis limits with minimum at -0.5 (to handle negative values properly)
    y_data_in_range = fft_real_normalized[np.logical_and(freq >= freq_min, freq <= freq_max)]
    y_max = np.max(y_data_in_range)
    y_min = np.min(y_data_in_range)
    
    # Ensure minimum y-axis shows at least -0.5 for proper baseline viewing
    y_min_display = min(-0.5, y_min * 1.1)
    y_max_display = y_max * 1.1
    
    ax.set_ylim(y_min_display, y_max_display)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()

def plot_individual_spectra(nmr_data_dict, save_individual=False):
    """
    Plot each spectrum individually
    
    Parameters:
    nmr_data_dict (dict): Dictionary with sample names as keys and (freq, real, mag) tuples as values
    save_individual (bool): Whether to save individual plots
    """
    for sample_name, (freq, fft_real, fft_mag) in nmr_data_dict.items():
        save_path = f"nmr_spectrum_{sample_name}.png" if save_individual else None
        plot_single_spectrum(freq, fft_real, fft_mag, sample_name, save_path)

def main():
    """
    Main function to process all NMR FFT files and create plots
    """
    # Define the path to the nmr folder
    nmr_folder = Path("/Users/charliefioriglio/Desktop/NMR/nmr")
    
    # Dictionary to store all NMR data
    nmr_data = {}
    
    # Find all .dat files in the nmr folder
    dat_files = list(nmr_folder.glob("*_fft.dat"))
    
    if not dat_files:
        print("No FFT .dat files found in the nmr folder!")
        return
    
    print(f"Found {len(dat_files)} FFT files:")
    for file in dat_files:
        print(f"  - {file.name}")
    
    # Read data from each file
    for file_path in dat_files:
        # Extract sample name from filename
        sample_name = file_path.stem.replace('_fft', '')
        
        print(f"\nReading {file_path.name}...")
        freq, fft_real, fft_mag = read_nmr_data(file_path)
        
        if freq is not None:
            nmr_data[sample_name] = (freq, fft_real, fft_mag)
            print(f"  ✓ Successfully loaded {len(freq)} data points")
            print(f"  ✓ Frequency range: {freq.min():.1f} - {freq.max():.1f} Hz")
        else:
            print(f"  ✗ Failed to load data from {file_path.name}")
    
    if not nmr_data:
        print("\nNo valid NMR data was loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(nmr_data)} NMR spectra")
    
    # Create plots
    print("\n" + "="*50)
    print("Creating NMR Spectrum Plots (Real Components)")
    print("="*50)
    
    # Plot individual spectra (real components only)
    print("\nCreating individual spectrum plots...")
    plot_individual_spectra(nmr_data, save_individual=True)
    
    print("\n✓ All plots completed!")

if __name__ == "__main__":
    main()
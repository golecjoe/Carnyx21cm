import numpy as np
from matplotlib.mlab import psd
from rtlsdr import RtlSdr
import os
import csv
from scipy.signal import welch


# Configurable parameters
ELEVATIONS = [80,70,60,50,40] #[20, 30, 40, 50, 60, 70, 80]  # in degrees
NFFT = 1024*1
NUM_SAMPLES = NFFT * 32
CENTER_FREQ = 1420.e6  # Hz
SAMPLE_RATE = 2.4e6   # Hz
GAIN = 36.6             # dB

OUTPUT_DIR = "skydip_data_040326_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def acquire_psd(sdr):
    samples = sdr.read_samples(NUM_SAMPLES)
    freqs, psd_vals = welch(samples, fs=sdr.sample_rate, nperseg=NFFT, return_onesided=False)
    freqs = freqs + sdr.center_freq  # shift to absolute frequency
    psd_db = 10 * np.log10(psd_vals)

    # Sort by frequency
    sort_idx = np.argsort(freqs)
    return freqs[sort_idx] / 1e6, psd_db[sort_idx]  # MHz

def save_to_csv(freqs, psd_db, elevation):
    filename = os.path.join(OUTPUT_DIR, f"psd_el{elevation}deg.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency_MHz", "PSD_dB"])
        for f, p in zip(freqs, psd_db):
            writer.writerow([f, p])
    print(f"Saved PSD data for elevation {elevation}° to {filename}")

def main():
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN

    for el in ELEVATIONS:
        input(f"Go to elevation {el} degrees and press Enter to take data...")
        freqs, psd_db = acquire_psd(sdr)
        save_to_csv(freqs, psd_db, el)

    sdr.close()
    print("All elevations complete.")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import psd
from rtlsdr import RtlSdr
import time

NFFT = 1024 * 1
NUM_SAMPLES_PER_SCAN = NFFT * 1024

def acquire_psd(sdr, integration_time_sec):
    total_psd = None
    total_iterations = 0

    start_time = time.time()
    while (time.time() - start_time) < integration_time_sec:
        samples = sdr.read_samples(NUM_SAMPLES_PER_SCAN)
        psd_vals, f = psd(samples, NFFT=NFFT)
        psd_db = 10 * np.log10(psd_vals)

        if total_psd is None:
            total_psd = psd_db
        else:
            total_psd += psd_db

        total_iterations += 1

    averaged_psd = total_psd / total_iterations
    fc = sdr.fc
    f_MHz = (fc + f * sdr.rs) / 1e6

    return f_MHz, averaged_psd

def plot_psd(frequencies, psd_values):
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd_values)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('PSD (dB)')
    plt.title('Integrated PSD')
    plt.grid(True)
    plt.show()

def main():
    integration_time_sec = float(input("Enter integration time in seconds: "))

    sdr = RtlSdr()
    sdr.rs = 2.4e6
    sdr.fc = 1420.5e6
    sdr.gain = 50

    print(f"Starting data acquisition for {integration_time_sec} seconds...")
    freqs, psd_vals = acquire_psd(sdr, integration_time_sec)
    print("Data acquisition complete. Plotting...")

    plot_psd(freqs, psd_vals)
    sdr.close()

    savearray = np.array([freqs,psd_vals])
    datadir = 'lab_data_040926_1/'
    np.savetxt(datadir+'resis_5_10sec.csv',savearray,delimiter=',')
    #np.savetxt(datadir+'dark_data4_3.csv',savearray,delimiter=',')

if __name__ == '__main__':
    main()

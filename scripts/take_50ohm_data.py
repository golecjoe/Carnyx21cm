#!/usr/bin/env python3
"""Acquire repeated integrations with a 50 ohm termination on the SDR input."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    acquire_integrated_psd,
    configure_sdr,
    ensure_output_dir,
    load_config,
    utc_now_iso,
    write_psd_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="../config/take_50ohm_data.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg["output"]["directory"])

    sdr = configure_sdr(cfg["sdr"])
    obs_cfg = cfg["observation"]

    load_label = obs_cfg.get("load_label", "50ohm_termination")
    n_scans = obs_cfg["num_scans"]
    center_freqs_hz = obs_cfg["center_freqs_hz"]
    settle_samples = int(obs_cfg.get("settle_samples", 0))

    try:
        for scan_idx in range(1, n_scans + 1):
            for freq_idx, center_freq_hz in enumerate(center_freqs_hz, start=1):
                print(
                    f"Starting scan {scan_idx}/{n_scans}, "
                    f"frequency {freq_idx}/{len(center_freqs_hz)} "
                    f"({center_freq_hz} Hz) for {load_label}..."
                )
                sdr.center_freq = float(center_freq_hz)
                if settle_samples > 0:
                    sdr.read_samples(settle_samples)

                freqs_hz, psd_linear, n_iter = acquire_integrated_psd(
                    sdr=sdr,
                    fft_size=obs_cfg["fft_size"],
                    samples_per_scan=obs_cfg["samples_per_scan"],
                    num_integrations=obs_cfg["num_integrations"],
                )

                ts = utc_now_iso().replace(":", "-")
                file_name = (
                    f"{ts}_{load_label.replace(' ', '_').lower()}"
                    f"_scan{scan_idx:03d}_fc{int(center_freq_hz)}.csv"
                )
                csv_path = Path(out_dir) / file_name
                meta = {
                    "task": "take_50ohm_data",
                    "timestamp_utc": utc_now_iso(),
                    "load_label": load_label,
                    "scan_index": scan_idx,
                    "scan_count": n_scans,
                    "frequency_index": freq_idx,
                    "frequency_count": len(center_freqs_hz),
                    "center_freq_hz": center_freq_hz,
                    "settle_samples": settle_samples,
                    "num_integrations": obs_cfg["num_integrations"],
                    "iterations": n_iter,
                    "sdr": cfg["sdr"],
                }
                write_psd_csv(csv_path, freqs_hz, psd_linear, meta)
                print(f"Saved: {csv_path}")
    finally:
        if hasattr(sdr, "set_bias_tee"):
            sdr.set_bias_tee(False)
        sdr.close()


if __name__ == "__main__":
    main()

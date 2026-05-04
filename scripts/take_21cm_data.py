#!/usr/bin/env python3
"""Acquire repeated integrations on target for 21cm line observations."""

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
    parser.add_argument("--config", default="config/take_21cm_data.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg["output"]["directory"])

    sdr = configure_sdr(cfg["sdr"])
    obs_cfg = cfg["observation"]

    target_name = obs_cfg.get("target_name", "unknown_target")
    n_scans = obs_cfg["num_scans"]

    try:
        for scan_idx in range(1, n_scans + 1):
            print(f"Starting scan {scan_idx}/{n_scans} for {target_name}...")
            freqs_hz, psd_linear, n_iter = acquire_integrated_psd(
                sdr=sdr,
                fft_size=obs_cfg["fft_size"],
                samples_per_scan=obs_cfg["samples_per_scan"],
                num_integrations=obs_cfg["num_integrations"],
            )

            ts = utc_now_iso().replace(":", "-")
            file_name = f"{ts}_{target_name.replace(' ', '_').lower()}_scan{scan_idx:03d}.csv"
            csv_path = Path(out_dir) / file_name
            meta = {
                "task": "take_21cm_data",
                "timestamp_utc": utc_now_iso(),
                "target_name": target_name,
                "scan_index": scan_idx,
                "scan_count": n_scans,
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

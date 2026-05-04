#!/usr/bin/env python3
"""Acquire spectra at multiple elevations for atmospheric opacity estimation."""

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
    parser.add_argument("--config", default="../config/sky_dip.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg["output"]["directory"])

    sdr = configure_sdr(cfg["sdr"])
    obs_cfg = cfg["observation"]

    try:
        for elevation_deg in obs_cfg["elevations_deg"]:
            input(f"Point horn to elevation {elevation_deg} deg and press Enter to capture... ")

            freqs_hz, psd_linear, n_iter = acquire_integrated_psd(
                sdr=sdr,
                fft_size=obs_cfg["fft_size"],
                samples_per_scan=obs_cfg["samples_per_scan"],
                num_integrations=obs_cfg["num_integrations"],
            )

            ts = utc_now_iso().replace(":", "-")
            file_name = f"{ts}_el{int(elevation_deg):02d}.csv"
            csv_path = Path(out_dir) / file_name
            meta = {
                "task": "sky_dip",
                "timestamp_utc": utc_now_iso(),
                "elevation_deg": elevation_deg,
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

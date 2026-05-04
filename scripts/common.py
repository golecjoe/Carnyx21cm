#!/usr/bin/env python3
"""Shared utilities for RTL-SDR 21cm observation scripts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import welch


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_sdr(sdr_cfg: dict[str, Any]) -> RtlSdr:
    sdr = RtlSdr()
    sdr.sample_rate = float(sdr_cfg["sample_rate_hz"])
    sdr.center_freq = float(sdr_cfg["center_freq_hz"])
    sdr.gain = sdr_cfg["gain_db"]
    bias_t = bool(sdr_cfg.get("bias_t", False))
    if hasattr(sdr, "set_bias_tee"):
        sdr.set_bias_tee(bias_t)

    return sdr


def acquire_integrated_psd(
    sdr: RtlSdr,
    fft_size: int,
    samples_per_scan: int,
    num_integrations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if num_integrations < 1:
        raise ValueError("num_integrations must be >= 1.")

    total_psd = None
    for _ in range(num_integrations):
        samples = sdr.read_samples(samples_per_scan)
        freqs, psd_vals = welch(
            samples,
            fs=sdr.sample_rate,
            nperseg=fft_size,
            return_onesided=False,
            scaling="density",
        )

        if total_psd is None:
            total_psd = psd_vals
        else:
            total_psd += psd_vals

    if total_psd is None:
        raise RuntimeError("No samples were collected.")

    avg_psd_linear = total_psd / num_integrations
    abs_freq_hz = freqs + sdr.center_freq

    sort_idx = np.argsort(abs_freq_hz)
    return abs_freq_hz[sort_idx], avg_psd_linear[sort_idx], num_integrations


def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_psd_csv(
    csv_path: Path,
    freqs_hz: np.ndarray,
    psd_linear: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# metadata_json", json.dumps(metadata, separators=(",", ":"))])
        writer.writerow(["frequency_hz", "frequency_mhz", "psd_linear_per_hz"])
        for f_hz, p_lin in zip(freqs_hz, psd_linear):
            writer.writerow([f"{f_hz:.3f}", f"{f_hz / 1e6:.9f}", f"{p_lin:.12e}"])

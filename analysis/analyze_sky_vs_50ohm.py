#!/usr/bin/env python3
"""Analyze sky observations relative to 50 ohm calibration data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_classes import FiftyOhmCalibrationData, SkyObservationData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_dir", help="Directory containing 50 ohm CSV files.")
    parser.add_argument("sky_dir", help="Directory containing sky observation CSV files.")
    parser.add_argument(
        "--output-dir",
        default="analysis/output/sky_vs_50ohm",
        help="Directory for ratio CSV files and the combined plot.",
    )
    parser.add_argument(
        "--stitched-bin-width-hz",
        type=float,
        default=None,
        help=(
            "Frequency bin width for the stitched ratio CSV. "
            "Defaults to the median native channel spacing."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_data = FiftyOhmCalibrationData(args.calibration_dir)
    sky_data = SkyObservationData(args.sky_dir)

    validate_center_frequencies(calibration_data, sky_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    output_csvs = []
    ratio_results = []

    for center_freq_hz in sky_data.center_freqs_hz:
        sky_avg = sky_data.get_average_for_tuning(center_freq_hz)
        calibration_avg = calibration_data.get_average_for_tuning(center_freq_hz)
        validate_frequency_grid(
            center_freq_hz,
            sky_avg.frequency_hz,
            calibration_avg.frequency_hz,
        )

        ratio = sky_avg.psd_linear_per_hz / calibration_avg.psd_linear_per_hz
        output_csv = output_dir / f"sky_over_50ohm_fc{center_freq_hz}.csv"
        write_ratio_csv(
            output_csv,
            center_freq_hz,
            sky_data,
            calibration_data,
            sky_avg.frequency_hz,
            sky_avg.psd_linear_per_hz,
            calibration_avg.psd_linear_per_hz,
            ratio,
        )
        output_csvs.append(output_csv)
        ratio_results.append(
            {
                "center_freq_hz": center_freq_hz,
                "frequency_hz": sky_avg.frequency_hz,
                "ratio": ratio,
            }
        )

        ax.plot(
            sky_avg.frequency_mhz,
            ratio,
            label=f"{center_freq_hz / 1e6:.3f} MHz",
        )

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("P_sky / P_50ohm")
    ax.set_title("Sky Spectrum Relative to 50 Ohm Calibration")
    ax.set_ylim(0.75,0.8)
    ax.legend(title="Center frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "sky_over_50ohm_all_tunings.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    stitched_bin_width_hz = choose_stitched_bin_width(
        ratio_results,
        args.stitched_bin_width_hz,
    )
    stitched_frequency_hz, stitched_ratio, stitched_counts = stitch_ratio_results(
        ratio_results,
        stitched_bin_width_hz,
    )
    stitched_csv = output_dir / "sky_over_50ohm_stitched.csv"
    write_stitched_ratio_csv(
        stitched_csv,
        stitched_bin_width_hz,
        sky_data,
        calibration_data,
        stitched_frequency_hz,
        stitched_ratio,
        stitched_counts,
    )
    stitched_plot_path = output_dir / "sky_over_50ohm_stitched.png"
    write_stitched_ratio_plot(
        stitched_plot_path,
        stitched_frequency_hz,
        stitched_ratio,
    )

    print(f"Saved plot: {plot_path}")
    print(f"Saved stitched CSV: {stitched_csv}")
    print(f"Saved stitched plot: {stitched_plot_path}")
    for output_csv in output_csvs:
        print(f"Saved CSV: {output_csv}")


def validate_center_frequencies(
    calibration_data: FiftyOhmCalibrationData,
    sky_data: SkyObservationData,
) -> None:
    if calibration_data.center_freqs_hz != sky_data.center_freqs_hz:
        raise ValueError(
            "Calibration and sky data must have the same center frequencies. "
            f"Calibration: {calibration_data.center_freqs_hz}; "
            f"Sky: {sky_data.center_freqs_hz}"
        )


def validate_frequency_grid(
    center_freq_hz: int,
    sky_frequency_hz: np.ndarray,
    calibration_frequency_hz: np.ndarray,
) -> None:
    if sky_frequency_hz.shape != calibration_frequency_hz.shape:
        raise ValueError(
            "Sky and calibration spectra have different frequency grid sizes "
            f"for center_freq_hz={center_freq_hz}."
        )
    if not np.allclose(
        sky_frequency_hz,
        calibration_frequency_hz,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            "Sky and calibration spectra have different frequency grids "
            f"for center_freq_hz={center_freq_hz}."
        )


def write_ratio_csv(
    output_csv: Path,
    center_freq_hz: int,
    sky_data: SkyObservationData,
    calibration_data: FiftyOhmCalibrationData,
    frequency_hz: np.ndarray,
    sky_psd: np.ndarray,
    calibration_psd: np.ndarray,
    ratio: np.ndarray,
) -> None:
    metadata = build_metadata(center_freq_hz, sky_data, calibration_data)

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["# metadata_json", json.dumps(metadata, separators=(",", ":"))])
        writer.writerow(
            [
                "frequency_hz",
                "frequency_mhz",
                "sky_psd_linear_per_hz",
                "calibration_psd_linear_per_hz",
                "sky_over_50ohm",
            ]
        )
        for freq_hz, sky_power, cal_power, power_ratio in zip(
            frequency_hz,
            sky_psd,
            calibration_psd,
            ratio,
        ):
            writer.writerow(
                [
                    f"{freq_hz:.3f}",
                    f"{freq_hz / 1e6:.9f}",
                    f"{sky_power:.12e}",
                    f"{cal_power:.12e}",
                    f"{power_ratio:.12e}",
                ]
            )


def choose_stitched_bin_width(
    ratio_results: list[dict[str, Any]],
    requested_bin_width_hz: float | None,
) -> float:
    if requested_bin_width_hz is not None:
        if requested_bin_width_hz <= 0:
            raise ValueError("--stitched-bin-width-hz must be positive.")
        return requested_bin_width_hz

    spacings = []
    for result in ratio_results:
        frequency_hz = result["frequency_hz"]
        if frequency_hz.size > 1:
            spacings.append(np.median(np.diff(frequency_hz)))

    if not spacings:
        raise ValueError("Cannot infer stitched bin width from single-point spectra.")

    return float(np.median(spacings))


def stitch_ratio_results(
    ratio_results: list[dict[str, Any]],
    bin_width_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.concatenate([result["frequency_hz"] for result in ratio_results])
    ratios = np.concatenate([result["ratio"] for result in ratio_results])

    sort_index = np.argsort(frequencies)
    frequencies = frequencies[sort_index]
    ratios = ratios[sort_index]

    bin_index = np.floor((frequencies - frequencies.min()) / bin_width_hz).astype(int)
    unique_bins = np.unique(bin_index)

    stitched_frequency_hz = np.empty(unique_bins.size)
    stitched_ratio = np.empty(unique_bins.size)
    stitched_counts = np.empty(unique_bins.size, dtype=int)

    for idx, bin_value in enumerate(unique_bins):
        in_bin = bin_index == bin_value
        stitched_frequency_hz[idx] = np.mean(frequencies[in_bin])
        stitched_ratio[idx] = np.mean(ratios[in_bin])
        stitched_counts[idx] = np.count_nonzero(in_bin)

    return stitched_frequency_hz, stitched_ratio, stitched_counts


def write_stitched_ratio_csv(
    output_csv: Path,
    bin_width_hz: float,
    sky_data: SkyObservationData,
    calibration_data: FiftyOhmCalibrationData,
    frequency_hz: np.ndarray,
    ratio: np.ndarray,
    counts: np.ndarray,
) -> None:
    metadata = {
        "task": "analyze_sky_vs_50ohm",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "product": "stitched_sky_over_50ohm",
        "ratio": "sky_psd_linear_per_hz / calibration_psd_linear_per_hz",
        "stitch_method": "mean ratio in uniform frequency bins",
        "stitched_bin_width_hz": bin_width_hz,
        "center_freqs_hz": sky_data.center_freqs_hz,
        "sky_data_dir": str(sky_data.data_dir),
        "calibration_data_dir": str(calibration_data.data_dir),
        "sky_scan_count": sky_data.n_scans,
        "calibration_scan_count": calibration_data.n_scans,
    }

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["# metadata_json", json.dumps(metadata, separators=(",", ":"))])
        writer.writerow(
            [
                "frequency_hz",
                "frequency_mhz",
                "sky_over_50ohm",
                "n_points_averaged",
            ]
        )
        for freq_hz, power_ratio, count in zip(frequency_hz, ratio, counts):
            writer.writerow(
                [
                    f"{freq_hz:.3f}",
                    f"{freq_hz / 1e6:.9f}",
                    f"{power_ratio:.12e}",
                    int(count),
                ]
            )


def write_stitched_ratio_plot(
    output_path: Path,
    frequency_hz: np.ndarray,
    ratio: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequency_hz / 1e6, ratio, color="black", linewidth=1.0)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("P_sky / P_50ohm")
    ax.set_title("Stitched Sky Spectrum Relative to 50 Ohm Calibration")
    ax.set_ylim(0.75,0.8)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_metadata(
    center_freq_hz: int,
    sky_data: SkyObservationData,
    calibration_data: FiftyOhmCalibrationData,
) -> dict[str, Any]:
    sky_scans = sky_data.get_scans_for_tuning(center_freq_hz)
    calibration_scans = calibration_data.get_scans_for_tuning(center_freq_hz)

    return {
        "task": "analyze_sky_vs_50ohm",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "center_freq_hz": center_freq_hz,
        "ratio": "sky_psd_linear_per_hz / calibration_psd_linear_per_hz",
        "sky_data_dir": str(sky_data.data_dir),
        "calibration_data_dir": str(calibration_data.data_dir),
        "sky_scan_count": len(sky_scans),
        "calibration_scan_count": len(calibration_scans),
        "sky_source_files": [str(scan.path) for scan in sky_scans],
        "calibration_source_files": [str(scan.path) for scan in calibration_scans],
        "sky_observations": [scan.metadata for scan in sky_scans],
        "calibration_observations": [scan.metadata for scan in calibration_scans],
    }


if __name__ == "__main__":
    main()

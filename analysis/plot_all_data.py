#!/usr/bin/env python3
"""Plot every spectrum scan in a directory and their per-tuning averages."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_classes import SpectrumDataSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing spectrum CSV files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory in which to save the generated plots.",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.csv",
        help="Glob used to select input files (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of saved PNG files (default: %(default)s).",
    )
    return parser.parse_args()


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (linear / Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_all_scans(data: SpectrumDataSet, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.colormaps["tab10"]

    for tuning_index, center_freq_hz in enumerate(data.center_freqs_hz):
        color = colors(tuning_index % colors.N)
        scans = data.get_scans_for_tuning(center_freq_hz)
        for scan_index, scan in enumerate(scans):
            ax.plot(
                scan.frequency_mhz,
                scan.psd_linear_per_hz,
                color=color,
                linewidth=0.8,
                alpha=0.55,
                label=(
                    f"{center_freq_hz / 1e6:.3f} MHz"
                    if scan_index == 0
                    else "_nolegend_"
                ),
            )

    style_axes(ax, f"All Spectrum Scans ({data.n_scans} scans)")
    ax.legend(title="Center frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_average_scans(data: SpectrumDataSet, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.colormaps["tab10"]

    for tuning_index, center_freq_hz in enumerate(data.center_freqs_hz):
        average = data.get_average_for_tuning(center_freq_hz)
        ax.plot(
            average.frequency_mhz,
            average.psd_linear_per_hz,
            color=colors(tuning_index % colors.N),
            linewidth=1.2,
            label=(
                f"{center_freq_hz / 1e6:.3f} MHz "
                f"({average.n_scans} scans)"
            ),
        )

    style_axes(ax, "Average Spectrum by Center Frequency")
    ax.legend(title="Center frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = SpectrumDataSet(
        Path(args.data_dir).expanduser(),
        file_pattern=args.file_pattern,
    )
    all_scans_path = output_dir / "all_scans.png"
    averages_path = output_dir / "average_scans.png"

    plot_all_scans(data, all_scans_path, args.dpi)
    plot_average_scans(data, averages_path, args.dpi)

    print(
        f"Loaded {data.n_scans} scans across {data.n_tunings} center frequencies."
    )
    print(f"Saved plot: {all_scans_path}")
    print(f"Saved plot: {averages_path}")


if __name__ == "__main__":
    main()

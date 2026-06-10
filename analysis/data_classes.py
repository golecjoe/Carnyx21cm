"""Data containers for calibration analysis."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_SCAN_RE = re.compile(r"_scan(?P<scan_index>\d+)")
_CENTER_FREQ_RE = re.compile(r"_fc(?P<center_freq_hz>\d+)")


@dataclass(frozen=True)
class PowerSpectrumScan:
    """One saved power spectrum scan."""

    path: Path
    frequency_hz: np.ndarray
    psd_linear_per_hz: np.ndarray
    metadata: dict[str, Any]
    scan_index: int | None = None
    center_freq_hz: int | None = None

    @property
    def frequency_mhz(self) -> np.ndarray:
        return self.frequency_hz / 1e6


@dataclass(frozen=True)
class TuningAverage:
    """Average spectrum for all scans at one SDR center-frequency tuning."""

    center_freq_hz: int
    scans: tuple[PowerSpectrumScan, ...]
    frequency_hz: np.ndarray
    psd_linear_per_hz: np.ndarray

    @property
    def frequency_mhz(self) -> np.ndarray:
        return self.frequency_hz / 1e6

    @property
    def n_scans(self) -> int:
        return len(self.scans)


class FiftyOhmCalibrationData:
    """Read, group, average, and stitch 50 ohm calibration scans.

    Parameters
    ----------
    data_dir
        Directory containing CSV files written by ``scripts/take_50ohm_data.py``.
    file_pattern
        Glob pattern used to find calibration CSV files.
    stitch_bin_width_hz
        Optional bin width for the stitched average. If omitted, tuning averages
        are concatenated and sorted by frequency without merging overlap regions.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_pattern: str = "*.csv",
        stitch_bin_width_hz: float | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.scans = tuple(self._read_all_scans())
        self.tunings = self._group_by_tuning(self.scans)
        self.tuning_averages = self._average_tunings(self.tunings)
        self.stitched_average = self.stitch_tuning_averages(stitch_bin_width_hz)

    @property
    def center_freqs_hz(self) -> tuple[int, ...]:
        return tuple(sorted(self.tunings))

    @property
    def n_scans(self) -> int:
        return len(self.scans)

    @property
    def n_tunings(self) -> int:
        return len(self.tunings)

    def get_scans_for_tuning(self, center_freq_hz: int) -> tuple[PowerSpectrumScan, ...]:
        return self.tunings[int(center_freq_hz)]

    def get_average_for_tuning(self, center_freq_hz: int) -> TuningAverage:
        return self.tuning_averages[int(center_freq_hz)]

    def stitch_tuning_averages(
        self,
        bin_width_hz: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a sorted spectrum made from all per-tuning averages.

        With ``bin_width_hz=None``, points from each averaged tuning are simply
        concatenated and sorted. With a positive bin width, points landing in the
        same frequency bin are averaged together.
        """

        if not self.tuning_averages:
            return np.array([]), np.array([])

        frequencies = np.concatenate(
            [avg.frequency_hz for avg in self.tuning_averages.values()]
        )
        powers = np.concatenate(
            [avg.psd_linear_per_hz for avg in self.tuning_averages.values()]
        )

        sort_index = np.argsort(frequencies)
        frequencies = frequencies[sort_index]
        powers = powers[sort_index]

        if bin_width_hz is None:
            return frequencies, powers
        if bin_width_hz <= 0:
            raise ValueError("bin_width_hz must be positive when provided.")

        return self._bin_average(frequencies, powers, float(bin_width_hz))

    def _read_all_scans(self) -> Iterable[PowerSpectrumScan]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory: {self.data_dir}")

        csv_paths = sorted(self.data_dir.glob(self.file_pattern))
        if not csv_paths:
            raise FileNotFoundError(
                f"No files matching {self.file_pattern!r} in {self.data_dir}"
            )

        for csv_path in csv_paths:
            yield self._read_scan(csv_path)

    def _read_scan(self, csv_path: Path) -> PowerSpectrumScan:
        metadata = self._read_metadata(csv_path)
        data = np.atleast_2d(np.loadtxt(csv_path, delimiter=",", skiprows=2))
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Expected at least three CSV data columns in {csv_path}")

        scan_index = self._scan_index(csv_path, metadata)
        center_freq_hz = self._center_freq_hz(csv_path, metadata)
        return PowerSpectrumScan(
            path=csv_path,
            frequency_hz=data[:, 0],
            psd_linear_per_hz=data[:, 2],
            metadata=metadata,
            scan_index=scan_index,
            center_freq_hz=center_freq_hz,
        )

    def _group_by_tuning(
        self,
        scans: tuple[PowerSpectrumScan, ...],
    ) -> dict[int, tuple[PowerSpectrumScan, ...]]:
        tunings: dict[int, list[PowerSpectrumScan]] = {}
        for scan in scans:
            if scan.center_freq_hz is None:
                raise ValueError(f"Could not infer center frequency for {scan.path}")
            tunings.setdefault(scan.center_freq_hz, []).append(scan)

        return {
            center_freq_hz: tuple(
                sorted(
                    tuning_scans,
                    key=lambda scan: (
                        scan.scan_index is None,
                        scan.scan_index if scan.scan_index is not None else 0,
                        scan.path.name,
                    ),
                )
            )
            for center_freq_hz, tuning_scans in sorted(tunings.items())
        }

    def _average_tunings(
        self,
        tunings: dict[int, tuple[PowerSpectrumScan, ...]],
    ) -> dict[int, TuningAverage]:
        averages = {}
        for center_freq_hz, scans in tunings.items():
            reference_frequency = scans[0].frequency_hz
            for scan in scans[1:]:
                if scan.frequency_hz.shape != reference_frequency.shape:
                    raise ValueError(
                        "Cannot average scans with different frequency grid sizes "
                        f"for center_freq_hz={center_freq_hz}."
                    )
                if not np.allclose(
                    scan.frequency_hz,
                    reference_frequency,
                    rtol=0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Cannot average scans with different frequency grids "
                        f"for center_freq_hz={center_freq_hz}."
                    )

            stacked_psd = np.vstack([scan.psd_linear_per_hz for scan in scans])
            averages[center_freq_hz] = TuningAverage(
                center_freq_hz=center_freq_hz,
                scans=scans,
                frequency_hz=reference_frequency.copy(),
                psd_linear_per_hz=np.mean(stacked_psd, axis=0),
            )

        return averages

    @staticmethod
    def _read_metadata(csv_path: Path) -> dict[str, Any]:
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            first_row = next(csv.reader(csv_file), None)

        if not first_row or first_row[0] != "# metadata_json":
            return {}

        try:
            return json.loads(first_row[1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not parse metadata JSON in {csv_path}") from exc

    @staticmethod
    def _scan_index(csv_path: Path, metadata: dict[str, Any]) -> int | None:
        if "scan_index" in metadata:
            return int(metadata["scan_index"])

        match = _SCAN_RE.search(csv_path.stem)
        if match is None:
            return None
        return int(match.group("scan_index"))

    @staticmethod
    def _center_freq_hz(csv_path: Path, metadata: dict[str, Any]) -> int | None:
        if "center_freq_hz" in metadata:
            return int(metadata["center_freq_hz"])
        if "center_freq_hz" in metadata.get("sdr", {}):
            return int(metadata["sdr"]["center_freq_hz"])

        match = _CENTER_FREQ_RE.search(csv_path.stem)
        if match is None:
            return None
        return int(match.group("center_freq_hz"))

    @staticmethod
    def _bin_average(
        frequencies_hz: np.ndarray,
        powers: np.ndarray,
        bin_width_hz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        bin_index = np.floor((frequencies_hz - frequencies_hz.min()) / bin_width_hz)
        unique_bins = np.unique(bin_index)

        binned_frequencies = np.empty(unique_bins.size)
        binned_powers = np.empty(unique_bins.size)
        for idx, bin_value in enumerate(unique_bins):
            in_bin = bin_index == bin_value
            binned_frequencies[idx] = np.mean(frequencies_hz[in_bin])
            binned_powers[idx] = np.mean(powers[in_bin])

        return binned_frequencies, binned_powers

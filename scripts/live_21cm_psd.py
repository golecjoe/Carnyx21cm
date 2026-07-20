#!/usr/bin/env python3
"""Display a continuously updating PSD from the 21 cm horn."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import acquire_integrated_psd, configure_sdr, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "live_21cm_psd.json"


def psd_to_db(psd_linear: np.ndarray) -> np.ndarray:
    """Convert linear PSD to dB/Hz without producing log-of-zero warnings."""
    return 10.0 * np.log10(np.maximum(psd_linear, np.finfo(float).tiny))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg = load_config(args.config)
    obs_cfg = cfg["observation"]
    plot_cfg = cfg.get("plot", {})

    settle_samples = int(obs_cfg.get("settle_samples", 0))
    refresh_interval_s = float(plot_cfg.get("refresh_interval_s", 0.05))
    if refresh_interval_s <= 0:
        raise ValueError("refresh_interval_s must be > 0.")

    sdr = configure_sdr(cfg["sdr"])
    fig = None
    try:
        plt.ion()
        fig, ax = plt.subplots()
        (line,) = ax.plot([], [], linewidth=1)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title(plot_cfg.get("title", "Live 21 cm Horn PSD"))
        ax.grid(bool(plot_cfg.get("grid", True)), alpha=0.3)

        if "y_min_db" in plot_cfg or "y_max_db" in plot_cfg:
            ax.set_ylim(
                plot_cfg.get("y_min_db", ax.get_ylim()[0]),
                plot_cfg.get("y_max_db", ax.get_ylim()[1]),
            )

        if settle_samples > 0:
            sdr.read_samples(settle_samples)

        while plt.fignum_exists(fig.number):
            freqs_hz, psd_linear, _ = acquire_integrated_psd(
                sdr=sdr,
                fft_size=obs_cfg["fft_size"],
                samples_per_scan=obs_cfg["samples_per_scan"],
                num_integrations=obs_cfg["num_integrations"],
            )
            freqs_mhz = freqs_hz / 1e6
            line.set_data(freqs_mhz, psd_to_db(psd_linear))
            ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
            if "y_min_db" not in plot_cfg and "y_max_db" not in plot_cfg:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(refresh_interval_s)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(sdr, "set_bias_tee"):
            sdr.set_bias_tee(False)
        sdr.close()
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()

21 cm RTL-SDR Pipeline Scaffold

Directory layout:
- scripts/common.py             Shared SDR + PSD acquisition helpers
- scripts/hot_cold_load.py      Hot/cold load capture task
- scripts/sky_dip.py            Sky dip capture task
- scripts/take_21cm_data.py     Repeated 21cm target captures
- scripts/take_50ohm_data.py    Repeated 50 ohm termination captures
- config/hot_cold_load.json     Config for hot/cold task
- config/sky_dip.json           Config for sky dip task
- config/take_21cm_data.json    Config for 21cm capture task
- config/take_50ohm_data.json   Config for 50 ohm capture task
- data/                         Output root for CSV files

Python dependencies:
- numpy
- scipy
- pyrtlsdr

Run examples:
- python3 scripts/hot_cold_load.py --config config/hot_cold_load.json
- python3 scripts/sky_dip.py --config config/sky_dip.json
- python3 scripts/take_21cm_data.py --config config/take_21cm_data.json
- python3 scripts/take_50ohm_data.py --config config/take_50ohm_data.json

Notes:
- Set `bias_t` true in config when powering the LNA from the RTL-SDR bias-T.
- `take_21cm_data.py` and `take_50ohm_data.py` sweep through `observation.center_freqs_hz` once per scan.
- `settle_samples` are discarded after each retune before integrating a spectrum.
- Each CSV includes a metadata JSON row followed by frequency + PSD columns.

# Red Pitaya MaRCoS Console (Python GUI)

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![GUI](https://img.shields.io/badge/GUI-PySide6%20%2B%20pyqtgraph-green)
![License](https://img.shields.io/badge/license-MIT-green)
![Built on](https://img.shields.io/badge/built%20on-MaRCoS%20%C2%B7%20MRI4all-orange)

A native **macOS** Python GUI for operating a **Red Pitaya SDRlab 122-16** running
[MaRCoS](https://github.com/vnegnev/marcos_extras) as a low-field MRI console —
styled after the [MRI4all](https://github.com/mri4all/console) console, which
targets Ubuntu only.

The GUI is the main project; the upstream MRI4all console is vendored under
[`mri4all_console/`](mri4all_console/) and its **real pulse-sequence builders are
reused** (not reimplemented) via [`seq_engine.py`](seq_engine.py).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rp_console.py            # launch the GUI
python rp_console.py --selftest # headless logic checks
python loopback_test.py --ip rp-f0d431.local --amp 0.2   # TX→RX PASS/FAIL test
```

See **[SETUP.md](SETUP.md)** for the board side — including the key trick for
running MaRCoS on **Red Pitaya OS 2.x** (load the bitstream with `fpgautil`, no
device-tree overlay; `marcos_server` talks to the FPGA via `/dev/mem`).

## What the GUI does

- **Built-in sequences** (self-contained): FID, Spin Echo, **TX→RX loopback test**,
  **noise scan**.
- **Calibrations**: **frequency sweep** (find resonance) and **RF amplitude
  calibration** — run an FID across the parameter and plot peak vs. value.
- **Console sequences**: the real `mri4all/console` pypulseq builders (`rf_se`,
  `se_1D`) run through flocra-pulseq → marcos, reused without a rewrite.
- Pulse-**sequence diagrams**, live **time-domain + FFT** plots, demo + real modes.
- MRI4all dark/amber theme, Home → Examination flow.

## Repository layout

```
.
├── rp_console.py          # the GUI (PySide6 + pyqtgraph)
├── seq_engine.py          # adapter that reuses the console's sequence builders
├── loopback_test.py       # standalone TX→RX bring-up test
├── requirements.txt
├── SETUP.md               # board / MaRCoS setup (OS 2.x bitstream-only)
├── assets/                # MRI4all branding
├── marcos_client/         # vendored marcos client (built-in sequences)
└── mri4all_console/       # vendored upstream mri4all/console (sequence builders)
```

`rp_console.py` and `seq_engine.py` auto-detect these folders, and also work in a
side-by-side dev layout (`gui/` next to `marcos/console`).

## Hardware status

The console + TX/RX chain are verified (loopback PASS; console `rf_se` runs on the
board). Real imaging additionally needs a tuned RF coil + sample, a B₀ magnet, and
the GPA gradient board (which unlocks `se_2D`/`tse` + image reconstruction).

## Credits

Built on [MaRCoS](https://github.com/vnegnev) (Vlad Negnevitsky et al.) and the
[MRI4all](https://mri4all.org) console (Tobias Block et al.). Upstream code under
`mri4all_console/` retains its original license.

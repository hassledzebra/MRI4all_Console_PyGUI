# Red Pitaya · MaRCoS mini-console (Python GUI)

A small native macOS GUI (PySide6 + pyqtgraph) that drives a Red Pitaya
running the MaRCoS server. It plays a single **pulse-and-acquire (FID)**
sequence and plots the time-domain signal and FFT spectrum.

It wraps the real `marcos_client` library located at
`../marcos/marcos_client`, so anything it does maps directly onto MaRCoS.

## Setup (one time)

```bash
cd gui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python rp_console.py            # launch the GUI
python rp_console.py --selftest # headless check of the sequence/DSP code
```

The GUI starts in **Demo mode** (synthetic FID) so you can use it with no
hardware attached. To talk to a real board:

1. Make sure `marcos_server` is running on the Red Pitaya.
2. Enter the board's host/IP (e.g. `rp-f0d431.local`) and port `11111`.
3. **Uncheck** "Demo mode" and press **Run acquisition**.

Connecting writes `../marcos/marcos_client/local_config.py` automatically.

## Controls

| Control | Meaning |
|---|---|
| Center freq | NCO / Larmor frequency in MHz (`lo_freq`) |
| RF pulse length | excitation pulse duration (µs) |
| RF amplitude | TX envelope amplitude, 0–1 (full scale) |
| RX dwell | sample spacing `rx_t` (µs); 3.125 µs is exact on the 122.88 MHz clock |
| # samples | number of complex samples to acquire |
| TX→RX deadtime | gap between the pulse and the receive window (µs) |

## Roadmap ideas

- Add spin-echo / multi-TR sequences (see `marcos_client/examples.py`).
- Frequency sweep / auto-find resonance.
- Save acquisitions to disk (`.npy` / HDF5).
- Averaging over repeated TRs for SNR.

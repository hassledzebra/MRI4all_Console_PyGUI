# Red Pitaya MaRCoS console — macOS GUI

A lightweight, native **macOS** console for a Red Pitaya **SDRlab 122-16** running
MaRCoS, styled after the MRI4all console. Built because the full `mri4all/console`
targets Ubuntu 22.04; this reuses the console's **sequence builders** but runs
on macOS with a small PySide6 GUI.

## The key insight: MaRCoS on Red Pitaya OS 2.x

The board's hardware requires a recent Red Pitaya OS (2.0+) to boot, but the stock
MaRCoS **device-tree overlay** is incompatible with OS 2.0 (`#address-cells` error
on `/fpga-full`). However, `marcos_server` accesses the FPGA via **`/dev/mem` at a
fixed physical address (`0x43C00000`)** — it does **not** need the overlay. So:

- Flash a current **Red Pitaya OS 2.x** image (e.g. `RedPitaya_OS_2.07-48_stable`).
- Load the marcos bitstream **without an overlay**:
  `fpgautil -b /root/marcos_fpga.bit.bin -f Full`
  (fpgautil deletes its `-b` source after loading — keep a persistent master in `/root`).
- Build/run `marcos_server` (gcc/cmake are on the board). A systemd unit makes it
  reboot-safe:
  ```
  [Service]
  ExecStartPre=/opt/redpitaya/bin/fpgautil -b /root/marcos_fpga.bit.bin -f Full
  ExecStart=/root/marcos_server
  ```

## GUI (`rp_console.py`)

PySide6 + pyqtgraph. Home → Examination flow, dark/amber MRI4all theme.
Sequences:

- **Built-in** (self-contained flodicts): FID, Spin Echo, **TX→RX loopback test**.
- **Console sequences** (`seq_engine.py`): the real `mri4all/console` pypulseq
  builders (`rf_se`, `se_1D`, …) → flocra-pulseq → marcos, reused without rebuild.

Run:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python rp_console.py            # GUI
python rp_console.py --selftest # headless logic checks
python loopback_test.py --ip rp-f0d431.local --amp 0.2   # PASS/FAIL TX→RX test
```

## Expected layout

`seq_engine.py` expects the console repo and marcos client as siblings:
```
Redpitaya/
  gui/            <- this folder (rp_console.py, seq_engine.py, …)
  marcos/
    console/      <- this repo (mri4all/console)
    marcos_client/  marcos_server/  marcos_extras/  flocra-pulseq/
```

## NumPy-2 compatibility patches

The console's bundled `external/marcos_client` predates NumPy 2.0 (which made
integer overflow raise instead of wrap). Patched for NumPy 2.x:
`marcompile.py` (`cl2ol` bit math) and `marmachine.py` (`insta`/`instb`) — bit
operations now done in Python ints. Gradient channels are stripped before running
(no GPA board), which also avoids the gpa-fhdo encoder overflow.

#!/usr/bin/env python3
"""
Red Pitaya / MaRCoS mini-console GUI.

A small PySide6 GUI that drives a Red Pitaya running the MaRCoS server.
It plays a simple pulse-and-acquire (FID) sequence and plots the
time-domain signal and its FFT spectrum.

It wraps the real `marcos_client` library:
    ex.ip_address / ex.port  ->  Experiment(lo_freq, rx_t)
    exp.add_flodict(flodict)
    rxd, msgs = exp.run()      # rxd['rx0'] is a complex numpy array

Demo mode synthesizes a realistic FID + noise so the GUI can be developed
and run with no hardware attached.

Usage:
    python rp_console.py            # launch the GUI
    python rp_console.py --selftest # exercise the DSP/sequence code headless
"""

import os
import sys
import argparse

import numpy as np

# --- Locate marcos_client (works in repo layout and dev layout) -------------
HERE = os.path.dirname(os.path.abspath(__file__))


def _locate_marcos_client():
    for c in (os.path.normpath(os.path.join(HERE, "..", "marcos", "marcos_client")),  # dev
              os.path.join(HERE, "marcos_client"),                                     # repo (vendored)
              os.path.join(HERE, "mri4all_console", "external", "marcos_client")):     # console-bundled
        if os.path.exists(os.path.join(c, "experiment.py")):
            return c
    return os.path.normpath(os.path.join(HERE, "..", "marcos", "marcos_client"))


MARCOS_CLIENT = _locate_marcos_client()


# ---------------------------------------------------------------------------
# Pure DSP / sequence helpers  (no Qt, no hardware -> unit-testable)
# ---------------------------------------------------------------------------

def build_fid_flodict(rf_len_us, rf_amp, rx_dwell_us, n_samples,
                      tx_gate_pre=1.0, tx_gate_post=1.0, deadtime_us=30.0,
                      rf_start_us=10.0):
    """Build a MaRCoS flo-dictionary for a single pulse-and-acquire (FID).

    Returns (flodict, meta) where meta carries derived timing for plotting.
    All times are in microseconds; tx0 values are complex in [-1-1j, 1+1j].
    """
    rf_end = rf_start_us + rf_len_us
    rx_start = rf_end + deadtime_us
    acq_len_us = n_samples * rx_dwell_us
    rx_end = rx_start + acq_len_us

    flodict = {
        # complex TX envelope: amplitude during the pulse, then 0
        "tx0": (np.array([rf_start_us, rf_end]),
                np.array([rf_amp + 0j, 0 + 0j])),
        # RF power-amp blanking gate, opened slightly before/after the pulse
        "tx_gate": (np.array([rf_start_us - tx_gate_pre, rf_end + tx_gate_post]),
                    np.array([1, 0])),
        # receive window
        "rx0_en": (np.array([rx_start, rx_end]),
                   np.array([1, 0])),
    }
    meta = {"rx_start_us": rx_start, "rx_end_us": rx_end,
            "acq_len_us": acq_len_us}
    return flodict, meta


def build_se_flodict(rf_len_us, rf_amp, rx_dwell_us, n_samples, te_ms=10.0,
                     tx_gate_pre=1.0, tx_gate_post=1.0, rf_start_us=10.0):
    """Build a spin-echo (90°–180°–echo) flo-dictionary.

    90° pulse, then a 180° refocusing pulse at TE/2, with the acquisition
    window centred on the echo at TE. The 180° pulse uses 2× the 90° duration
    (≈2× flip) at the same amplitude. Times in µs; tx0 values complex.
    """
    te_us = te_ms * 1e3
    c90 = rf_start_us + rf_len_us / 2.0            # 90° pulse centre
    rf90_s, rf90_e = rf_start_us, rf_start_us + rf_len_us
    c180 = c90 + te_us / 2.0                        # 180° centre at TE/2
    rf180_len = 2.0 * rf_len_us
    rf180_s, rf180_e = c180 - rf180_len / 2.0, c180 + rf180_len / 2.0
    echo_c = c90 + te_us                            # echo centre at TE
    acq_len_us = n_samples * rx_dwell_us
    rx_s, rx_e = echo_c - acq_len_us / 2.0, echo_c + acq_len_us / 2.0

    flodict = {
        "tx0": (np.array([rf90_s, rf90_e, rf180_s, rf180_e]),
                np.array([rf_amp + 0j, 0 + 0j, rf_amp + 0j, 0 + 0j])),
        "tx_gate": (np.array([rf90_s - tx_gate_pre, rf90_e + tx_gate_post,
                              rf180_s - tx_gate_pre, rf180_e + tx_gate_post]),
                    np.array([1, 0, 1, 0])),
        "rx0_en": (np.array([rx_s, rx_e]), np.array([1, 0])),
    }
    meta = {"rx_start_us": rx_s, "rx_end_us": rx_e, "acq_len_us": acq_len_us,
            "echo_us": echo_c}
    return flodict, meta


def build_loopback_flodict(rf_len_us, rf_amp, rx_dwell_us, n_samples,
                           rx_start_us=10.0):
    """TX→RX cable-loopback test: a TX pulse captured by an OVERLAPPING RX
    window. With the TX output cabled to the RX input, the receiver samples
    the transmitted signal directly — a quick proof the TX/RX chain works.

    The pulse runs through the middle ~80% of the acquisition so you see the
    receiver go from noise → strong signal → noise as TX turns on and off.
    """
    acq_len_us = n_samples * rx_dwell_us
    rx_end = rx_start_us + acq_len_us
    tx_s = rx_start_us + 0.1 * acq_len_us
    tx_e = rx_end - 0.1 * acq_len_us
    flodict = {
        "tx0": (np.array([tx_s, tx_e]), np.array([rf_amp + 0j, 0 + 0j])),
        "tx_gate": (np.array([tx_s - 1.0, tx_e + 1.0]), np.array([1, 0])),
        "rx0_en": (np.array([rx_start_us, rx_end]), np.array([1, 0])),
    }
    meta = {"rx_start_us": rx_start_us, "rx_end_us": rx_end,
            "acq_len_us": acq_len_us, "tx_window_us": (tx_s, tx_e)}
    return flodict, meta


def build_sequence(params):
    """Dispatch to the selected sequence builder. params['sequence'] in
    {'fid','se','loopback'}. Returns (flodict, meta)."""
    seq = params.get("sequence")
    if seq == "se":
        return build_se_flodict(params["rf_len_us"], params["rf_amp"],
                                params["rx_dwell_us"], params["n_samples"],
                                te_ms=params.get("te_ms", 10.0))
    if seq == "loopback":
        return build_loopback_flodict(params["rf_len_us"], params["rf_amp"],
                                      params["rx_dwell_us"], params["n_samples"])
    return build_fid_flodict(params["rf_len_us"], params["rf_amp"],
                             params["rx_dwell_us"], params["n_samples"],
                             deadtime_us=params["deadtime_us"])


def synth_fid(n_samples, rx_dwell_us, offset_khz=2.0, t2_ms=8.0,
              amp=4e-4, noise=4e-5, seed=0):
    """Synthesize a decaying complex FID + Gaussian noise (demo mode)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * rx_dwell_us * 1e-6           # seconds
    decay = np.exp(-t / (t2_ms * 1e-3))
    phase = np.exp(2j * np.pi * (offset_khz * 1e3) * t)
    sig = amp * decay * phase
    sig = sig + noise * (rng.standard_normal(n_samples) +
                         1j * rng.standard_normal(n_samples))
    return sig.astype(np.complex128)


def synth_loopback(n_samples, amp=0.3, noise=4e-5, seed=1):
    """Synthesize a TX→RX loopback capture: ~constant amplitude while TX is on
    (middle ~80% of the window), noise at the edges (demo mode)."""
    rng = np.random.default_rng(seed)
    sig = noise * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    a, b = int(0.1 * n_samples), int(0.9 * n_samples)
    sig[a:b] += amp * np.exp(1j * 0.3)        # steady transmitted level
    return sig.astype(np.complex128)


def spectrum(data, rx_dwell_us):
    """Return (freq_khz, magnitude) of the centered FFT of complex data."""
    n = len(data)
    if n == 0:
        return np.array([]), np.array([])
    spec = np.fft.fftshift(np.fft.fft(data))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=rx_dwell_us * 1e-6))  # Hz
    return freq / 1e3, np.abs(spec)


def sequence_waveforms(flodict, t_end_us=None):
    """Expand a flodict's (times, values) change-points into step waveforms
    suitable for a pulse-sequence diagram. Returns {channel: (t_us, value)}.

    Each flodict entry is (times, values) where the channel takes value[i] at
    time[i] (µs) and holds it until the next change. Complex TX is shown as |amp|.
    """
    arrs = {k: (np.asarray(t, float), np.asarray(v)) for k, (t, v) in flodict.items()}
    if t_end_us is None:
        last = max((t[-1] for t, _ in arrs.values() if len(t)), default=0.0)
        t_end_us = last * 1.05 if last else 1.0
    out = {}
    for ch, (t, v) in arrs.items():
        vv = np.abs(v) if np.iscomplexobj(v) else v.astype(float)
        ts, vs = [0.0], [0.0]              # start held at 0
        for i in range(len(t)):
            ts += [float(t[i]), float(t[i])]
            vs += [vs[-1], float(vv[i])]   # hold, then step to new value
        ts.append(float(t_end_us)); vs.append(vs[-1])
        out[ch] = (np.array(ts), np.array(vs))
    return out


def run_console_sequence(params, demo=False):
    """Run a real mri4all/console sequence via the bundled sequence engine.

    params['sequence'] is 'console:<name>'; params['console_params'] holds the
    sequence's own inputs (TE/TR/NSA/…). Returns the same dict shape as
    run_experiment so the GUI plots it identically.
    """
    import seq_engine
    name = params["sequence"].split(":", 1)[1]
    cparams = params.get("console_params", {})
    if demo:
        iq = synth_fid(512, 3.125)        # no board: just show structure
        rx_t, msgs, flo = 3.125, f"DEMO: console '{name}' (synthetic).", None
    else:
        res = seq_engine.run_seq(name, cparams, params["ip"], params["freq_mhz"],
                                 port=params.get("port", 11111))
        iq, rx_t, flo = res["iq"], res["rx_t_us"], res["flodict"]
        msgs = f"console '{name}': {res['readouts']} readouts. {res['msgs'][:80]}"
    n = len(iq)
    t_ms = np.arange(n) * rx_t * 1e-3
    f_khz, mag = spectrum(iq, rx_t)
    return {"t_ms": t_ms, "iq": iq, "f_khz": f_khz, "mag": mag, "msgs": msgs,
            "flodict": flo, "meta": {}}


def run_experiment(params, demo=False):
    """Run one acquisition. Returns dict with time/freq arrays + msgs.

    params: dict with keys freq_mhz, rf_len_us, rf_amp, rx_dwell_us,
            n_samples, deadtime_us, ip, port, sequence.
    demo:   if True, synthesize data instead of touching hardware.
    """
    if str(params.get("sequence", "")).startswith("console:"):
        return run_console_sequence(params, demo)

    flodict, meta = build_sequence(params)

    if demo:
        if params.get("sequence") == "loopback":
            data = synth_loopback(params["n_samples"], amp=0.6 * params["rf_amp"])
            msgs = "DEMO MODE: synthetic TX→RX loopback (no hardware contacted)."
        else:
            data = synth_fid(params["n_samples"], params["rx_dwell_us"])
            msgs = "DEMO MODE: synthetic FID (no hardware contacted)."
    else:
        import socket as _socket
        ensure_local_config(params["ip"], params["port"])
        if MARCOS_CLIENT not in sys.path:
            sys.path.insert(0, MARCOS_CLIENT)
        import experiment as ex  # imported here so config is in place first
        # Patch the module-level connection globals (read in Experiment.__init__)
        ex.ip_address = params["ip"]
        ex.port = int(params["port"])
        # Bound every socket op so a busy/hung server (marcos is single-client)
        # raises instead of freezing the scan forever.
        _old_to = _socket.getdefaulttimeout()
        _socket.setdefaulttimeout(float(params.get("timeout_s", 20.0)))
        exp = None
        try:
            exp = ex.Experiment(lo_freq=params["freq_mhz"],
                                rx_t=params["rx_dwell_us"],
                                init_gpa=False, halt_and_reset=True)
            exp.add_flodict(flodict)
            rxd, server_msgs = exp.run()
        except (_socket.timeout, OSError) as e:
            raise RuntimeError(
                f"No response from {params['ip']}:{params['port']} "
                f"({type(e).__name__}). Is marcos_server running, and not "
                f"already in use by another client (GUI + CLI can't share it)?")
        finally:
            _socket.setdefaulttimeout(_old_to)
            # CRITICAL: marcos_server is single-client — close the socket so the
            # next scan can connect (otherwise back-to-back scans fail).
            if exp is not None:
                try:
                    exp._s.close()
                except Exception:
                    pass
        data = rxd.get("rx0", np.array([], dtype=np.complex128))
        msgs = "\n".join(str(m) for m in server_msgs) if server_msgs else "OK"

    n = len(data)
    t_ms = np.arange(n) * params["rx_dwell_us"] * 1e-3
    f_khz, mag = spectrum(data, params["rx_dwell_us"])
    return {"t_ms": t_ms, "iq": data, "f_khz": f_khz, "mag": mag,
            "msgs": msgs, "meta": meta}


# ---------------------------------------------------------------------------
# local_config.py management
# ---------------------------------------------------------------------------

def ensure_local_config(ip, port, fpga_clk_freq_MHz=122.88, grad_board="gpa-fhdo"):
    """Write marcos_client/local_config.py with the given IP/port if needed."""
    path = os.path.join(MARCOS_CLIENT, "local_config.py")
    content = (
        "## Auto-generated by rp_console.py\n"
        f'ip_address = "{ip}"\n'
        f"port = {int(port)}\n"
        f"fpga_clk_freq_MHz = {fpga_clk_freq_MHz}\n"
        f'grad_board = "{grad_board}"\n'
        "gpa_fhdo_current_per_volt = 2.5\n"
    )
    try:
        if os.path.exists(path) and open(path).read() == content:
            return path
        with open(path, "w") as f:
            f.write(content)
    except OSError as e:
        raise RuntimeError(f"Could not write {path}: {e}")
    return path


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

# --- MRI4all visual identity (palette extracted from mri4all/console) -------
MRI4ALL_BG      = "#040919"   # near-black navy background
MRI4ALL_PANEL   = "#262C44"   # panel / section blue
MRI4ALL_ACCENT  = "#E0A526"   # signature amber/gold
MRI4ALL_DIM     = "#424d76"   # dimmed text / borders
MRI4ALL_FG       = "#FFFFFF"

MRI4ALL_QSS = f"""
QWidget {{ background-color: {MRI4ALL_BG}; color: {MRI4ALL_FG}; font-size: 15px; }}
QFrame#header {{ background-color: {MRI4ALL_PANEL}; }}
QFrame#panel  {{ background-color: {MRI4ALL_PANEL}; border-radius: 8px; }}
QLabel#title  {{ color: {MRI4ALL_FG}; font-size: 22px; font-weight: 600; }}
QLabel#sysinfo {{ color: {MRI4ALL_ACCENT}; font-size: 14px; }}
QLabel#section {{ color: {MRI4ALL_ACCENT}; font-size: 13px; font-weight: 600;
                  letter-spacing: 1px; }}
QLabel {{ background: transparent; }}
QPushButton {{ color: {MRI4ALL_FG}; background-color: {MRI4ALL_PANEL};
               border: 1px solid {MRI4ALL_DIM}; border-radius: 6px; padding: 8px 16px; }}
QPushButton:hover {{ background-color: {MRI4ALL_ACCENT}; color: {MRI4ALL_BG}; }}
QPushButton[type="primary"] {{ background-color: {MRI4ALL_ACCENT}; color: {MRI4ALL_BG};
                               font-weight: 600; border: none; padding: 12px 28px; font-size: 17px; }}
QPushButton[type="primary"]:hover {{ background-color: #f2b945; }}
QPushButton:disabled {{ color: {MRI4ALL_DIM}; background-color: {MRI4ALL_PANEL}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background-color: #00000033; color: {MRI4ALL_FG};
    border: 1px solid {MRI4ALL_DIM}; border-radius: 4px; padding: 4px; }}
QCheckBox {{ background: transparent; }}
QCheckBox::indicator:checked {{ background-color: {MRI4ALL_ACCENT}; border: 1px solid {MRI4ALL_ACCENT}; }}
QSplitter::handle {{ background-color: {MRI4ALL_BG}; }}
"""


def launch_gui(smoke=False):
    from PySide6 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg

    ASSETS = os.path.join(HERE, "assets")
    # dark plots to match the console
    pg.setConfigOptions(antialias=True, background=MRI4ALL_BG, foreground=MRI4ALL_FG)

    class Worker(QtCore.QThread):
        done = QtCore.Signal(dict)
        failed = QtCore.Signal(str)

        def __init__(self, params, demo):
            super().__init__()
            self.params, self.demo = params, demo

        def run(self):
            try:
                self.done.emit(run_experiment(self.params, demo=self.demo))
            except Exception as e:  # surface any hardware/network error to UI
                self.failed.emit(f"{type(e).__name__}: {e}")

    def asset(name):
        return os.path.join(ASSETS, name)

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MRI4ALL")
            icon = asset("mri4all_icon.png")
            if os.path.exists(icon):
                self.setWindowIcon(QtGui.QIcon(icon))
            self.resize(1180, 720)
            self.worker = None

            self.stack = QtWidgets.QStackedWidget()
            self.stack.addWidget(self._build_home())          # page 0
            self.stack.addWidget(self._build_examination())   # page 1
            self.setCentralWidget(self.stack)

        # ---- shared header bar (logo + title + system info) ----
        def _header(self, subtitle):
            hdr = QtWidgets.QFrame(); hdr.setObjectName("header"); hdr.setFixedHeight(72)
            h = QtWidgets.QHBoxLayout(hdr); h.setContentsMargins(18, 8, 18, 8)
            logo = QtWidgets.QLabel()
            lp = asset("mri4all_logo.png")
            if os.path.exists(lp):
                logo.setPixmap(QtGui.QPixmap(lp).scaledToHeight(44, QtCore.Qt.SmoothTransformation))
            h.addWidget(logo)
            t = QtWidgets.QLabel(subtitle); t.setObjectName("title")
            h.addWidget(t); h.addStretch()
            sysinfo = QtWidgets.QLabel("Zeugmatron Z1  ·  SDRlab 122-16  ·  MaRCoS")
            sysinfo.setObjectName("sysinfo")
            h.addWidget(sysinfo)
            return hdr

        # ---- page 0: landing screen (mimics registration/home) ----
        def _build_home(self):
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
            v.addWidget(self._header("Console"))
            body = QtWidgets.QWidget()       # solid navy background (no tiling)
            bl = QtWidgets.QVBoxLayout(body); bl.addStretch()
            # single, centered logo
            logo = QtWidgets.QLabel(); logo.setAlignment(QtCore.Qt.AlignCenter)
            lp = asset("mri4all_logo.png")
            if os.path.exists(lp):
                logo.setPixmap(QtGui.QPixmap(lp).scaledToHeight(120, QtCore.Qt.SmoothTransformation))
            bl.addWidget(logo); bl.addSpacing(24)
            big = QtWidgets.QLabel("MRI4ALL Console")
            big.setStyleSheet(f"font-size: 40px; font-weight: 700; color: {MRI4ALL_FG}; background: transparent;")
            big.setAlignment(QtCore.Qt.AlignCenter); bl.addWidget(big)
            sub = QtWidgets.QLabel("Red Pitaya 122-16  ·  MaRCoS backend")
            sub.setStyleSheet(f"font-size: 18px; color: {MRI4ALL_ACCENT}; background: transparent;")
            sub.setAlignment(QtCore.Qt.AlignCenter); bl.addWidget(sub)
            bl.addSpacing(34)
            start = QtWidgets.QPushButton("Start Examination")
            start.setProperty("type", "primary"); start.setFixedWidth(280)
            start.clicked.connect(lambda: self.stack.setCurrentIndex(1))
            row = QtWidgets.QHBoxLayout(); row.addStretch(); row.addWidget(start); row.addStretch()
            bl.addLayout(row); bl.addStretch()
            v.addWidget(body)
            return page

        # ---- page 1: examination (acquisition + plots) ----
        def _build_examination(self):
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
            v.addWidget(self._header("Examination"))

            content = QtWidgets.QWidget(); cl = QtWidgets.QHBoxLayout(content)
            cl.setContentsMargins(14, 14, 14, 14); cl.setSpacing(14)

            # left: sequence parameters panel
            panel = QtWidgets.QFrame(); panel.setObjectName("panel"); panel.setFixedWidth(330)
            pl = QtWidgets.QVBoxLayout(panel); pl.setContentsMargins(16, 16, 16, 16)
            pl.addWidget(self._sectionlabel("CONNECTION"))
            form = QtWidgets.QFormLayout(); form.setLabelAlignment(QtCore.Qt.AlignRight)
            self.ip = QtWidgets.QLineEdit("rp-f0d431.local")
            self.port = QtWidgets.QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(11111)
            self.freq = QtWidgets.QDoubleSpinBox(); self.freq.setRange(0.001, 62.0); self.freq.setDecimals(4); self.freq.setValue(2.0); self.freq.setSuffix(" MHz")
            self.rf_len = QtWidgets.QDoubleSpinBox(); self.rf_len.setRange(0.1, 1000); self.rf_len.setValue(10.0); self.rf_len.setSuffix(" µs")
            self.rf_amp = QtWidgets.QDoubleSpinBox(); self.rf_amp.setRange(0.0, 1.0); self.rf_amp.setSingleStep(0.05); self.rf_amp.setValue(0.5)
            self.dwell = QtWidgets.QDoubleSpinBox(); self.dwell.setRange(0.05, 100); self.dwell.setDecimals(3); self.dwell.setValue(3.125); self.dwell.setSuffix(" µs")
            self.nsamp = QtWidgets.QSpinBox(); self.nsamp.setRange(8, 100000); self.nsamp.setValue(512)
            self.dead = QtWidgets.QDoubleSpinBox(); self.dead.setRange(0, 1000); self.dead.setValue(30.0); self.dead.setSuffix(" µs")
            self.te = QtWidgets.QDoubleSpinBox(); self.te.setRange(0.1, 1000); self.te.setDecimals(2); self.te.setValue(10.0); self.te.setSuffix(" ms")
            self.seq_combo = QtWidgets.QComboBox()
            self.seq_combo.addItem("FID (pulse-acquire)", "fid")
            self.seq_combo.addItem("Spin Echo (90°–180°)", "se")
            self.seq_combo.addItem("Loopback test (TX→RX)", "loopback")
            # real mri4all/console sequences (reused via the bundled engine)
            try:
                import seq_engine
                if seq_engine.available():
                    self.seq_combo.insertSeparator(self.seq_combo.count())
                    for key in ("rf_se", "se_1D"):
                        self.seq_combo.addItem(seq_engine.list_sequences()[key] + " · console",
                                               f"console:{key}")
            except Exception:
                pass
            self.demo = QtWidgets.QCheckBox("Demo mode (no hardware)"); self.demo.setChecked(True)
            form.addRow("Host / IP", self.ip); form.addRow("Port", self.port)
            pl.addLayout(form)
            pl.addSpacing(8); pl.addWidget(self._sectionlabel("SEQUENCE"))
            form2 = QtWidgets.QFormLayout(); form2.setLabelAlignment(QtCore.Qt.AlignRight)
            self.dead_label = QtWidgets.QLabel("TX→RX deadtime")
            self.te_label = QtWidgets.QLabel("Echo time (TE)")
            form2.addRow("Sequence", self.seq_combo)
            for label, w in [("Center freq", self.freq), ("RF length", self.rf_len),
                             ("RF amplitude", self.rf_amp), ("RX dwell", self.dwell),
                             ("# samples", self.nsamp)]:
                form2.addRow(label, w)
            form2.addRow(self.dead_label, self.dead)
            form2.addRow(self.te_label, self.te)
            pl.addLayout(form2)
            # dynamic parameter form for console sequences (TE/TR/NSA/…)
            self.console_box = QtWidgets.QWidget()
            self.console_form = QtWidgets.QFormLayout(self.console_box)
            self.console_form.setLabelAlignment(QtCore.Qt.AlignRight)
            self.console_form.setContentsMargins(0, 0, 0, 0)
            self.console_widgets = {}
            pl.addWidget(self.console_box)
            self.console_box.setVisible(False)
            self.seq_combo.currentIndexChanged.connect(self._on_seq_changed)
            pl.addWidget(self.demo)
            pl.addSpacing(8)
            self.run_btn = QtWidgets.QPushButton("▶  Run Scan")
            self.run_btn.setProperty("type", "primary")
            self.run_btn.clicked.connect(self.on_run)
            pl.addWidget(self.run_btn)
            self.preview_btn = QtWidgets.QPushButton("⊞  Preview Sequence")
            self.preview_btn.clicked.connect(self.on_preview)
            pl.addWidget(self.preview_btn)
            back = QtWidgets.QPushButton("‹ Home"); back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
            pl.addWidget(back)
            pl.addStretch()
            self.status = QtWidgets.QPlainTextEdit(); self.status.setReadOnly(True)
            self.status.setFixedHeight(120); self.status.setPlaceholderText("Status / server messages…")
            pl.addWidget(self._sectionlabel("LOG")); pl.addWidget(self.status)
            cl.addWidget(panel)

            # right: tabs — Acquisition (data) and Sequence (pulse diagram)
            tabs = QtWidgets.QTabWidget()

            # -- Acquisition tab --
            self.p_time = pg.PlotWidget(title="Time domain")
            self.p_time.addLegend(); self.p_time.setLabel("bottom", "Time", "ms")
            self.c_i = self.p_time.plot(pen=pg.mkPen("#5b8ff9"), name="I")
            self.c_q = self.p_time.plot(pen=pg.mkPen("#9270CA"), name="Q")
            self.c_m = self.p_time.plot(pen=pg.mkPen(MRI4ALL_ACCENT, width=2), name="|signal|")
            self.p_spec = pg.PlotWidget(title="Spectrum (FFT magnitude)")
            self.p_spec.setLabel("bottom", "Frequency offset", "kHz")
            self.c_s = self.p_spec.plot(pen=pg.mkPen(MRI4ALL_ACCENT, width=2))
            acq = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            acq.addWidget(self.p_time); acq.addWidget(self.p_spec)
            tabs.addTab(acq, "Acquisition")

            # -- Sequence tab: stacked step plots (TX, TX gate, RX window) --
            self.seq_rows = {}
            seqw = QtWidgets.QWidget(); seqv = QtWidgets.QVBoxLayout(seqw); seqv.setSpacing(2)
            for ch, label, color in [("tx0", "RF (TX) |amp|", MRI4ALL_ACCENT),
                                     ("tx_gate", "TX gate", "#5b8ff9"),
                                     ("rx0_en", "RX window", "#46c46a")]:
                pw = pg.PlotWidget(); pw.setMaximumHeight(170)
                pw.setLabel("left", label)
                pw.showGrid(x=True, y=False, alpha=0.2)
                self.seq_rows[ch] = pw.plot(pen=pg.mkPen(color, width=2), stepMode=False)
                if ch == "rx0_en":
                    pw.setLabel("bottom", "Time (µs)")   # plain label, no SI auto-scaling
                seqv.addWidget(pw)
            tabs.addTab(seqw, "Sequence")
            self.tabs = tabs
            cl.addWidget(tabs, 1)

            self._on_seq_changed()   # set initial TE/deadtime visibility + draw sequence
            v.addWidget(content)
            return page

        def _sectionlabel(self, text):
            lab = QtWidgets.QLabel(text); lab.setObjectName("section"); return lab

        def log(self, msg):
            self.status.appendPlainText(msg)

        @staticmethod
        def _is_console(seq):
            return isinstance(seq, str) and seq.startswith("console:")

        def _params(self):
            seq = self.seq_combo.currentData()
            p = dict(ip=self.ip.text().strip(), port=self.port.value(),
                     freq_mhz=self.freq.value(), rf_len_us=self.rf_len.value(),
                     rf_amp=self.rf_amp.value(), rx_dwell_us=self.dwell.value(),
                     n_samples=self.nsamp.value(), deadtime_us=self.dead.value(),
                     sequence=seq, te_ms=self.te.value())
            if self._is_console(seq):
                p["console_params"] = {k: g() for k, g in self.console_widgets.items()}
            return p

        def _populate_console_form(self, name):
            import seq_engine
            from PySide6 import QtWidgets as W
            while self.console_form.rowCount():
                self.console_form.removeRow(0)
            self.console_widgets = {}
            for k, v in seq_engine.defaults_for(name).items():
                if isinstance(v, bool):
                    w = W.QCheckBox(); w.setChecked(v); g = w.isChecked
                elif isinstance(v, int):
                    w = W.QSpinBox(); w.setRange(0, 1000000); w.setValue(v); g = w.value
                elif isinstance(v, float):
                    w = W.QDoubleSpinBox(); w.setRange(0, 1e6); w.setDecimals(3); w.setValue(v); g = w.value
                else:
                    w = W.QLineEdit(str(v)); g = w.text
                self.console_form.addRow(k, w)
                self.console_widgets[k] = g

        def _on_seq_changed(self, *_):
            seq = self.seq_combo.currentData()
            is_con = self._is_console(seq)
            self.te_label.setVisible(seq == "se"); self.te.setVisible(seq == "se")
            self.dead_label.setVisible(seq == "fid"); self.dead.setVisible(seq == "fid")
            # built-in pulse params don't apply to console sequences (they bring their own)
            for w in (self.rf_len, self.rf_amp, self.dwell, self.nsamp):
                w.setEnabled(not is_con)
            self.console_box.setVisible(is_con)
            if is_con:
                self._populate_console_form(seq.split(":", 1)[1])
            elif hasattr(self, "seq_rows"):
                self._draw_sequence(self._params())

        def _draw_flodict(self, flodict, t_end=None):
            wf = sequence_waveforms(flodict, t_end_us=t_end)
            for ch, curve in self.seq_rows.items():
                if ch in wf:
                    t, y = wf[ch]; curve.setData(t, y)
                else:
                    curve.setData([], [])

        def _draw_sequence(self, params):
            flodict, meta = build_sequence(params)
            self._draw_flodict(flodict, t_end=meta["rx_end_us"] * 1.05)

        def on_preview(self):
            if self._is_console(self.seq_combo.currentData()):
                self.log("Console sequence: press Run Scan — its diagram is drawn "
                         "from the compiled waveforms after the scan.")
                return
            self._draw_sequence(self._params())
            self.tabs.setCurrentIndex(1)   # show Sequence tab
            self.log("Sequence preview updated.")

        def on_run(self):
            if self.worker is not None and self.worker.isRunning():
                self.log("A scan is already running — please wait.")
                return
            params = self._params()
            if not self._is_console(params.get("sequence", "")):
                self._draw_sequence(params)
            demo = self.demo.isChecked()
            self.run_btn.setEnabled(False)
            self.log(("DEMO scan…" if demo else f"Connecting to {params['ip']}:{params['port']} …"))
            self.worker = Worker(params, demo)
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            # Backstop: whatever happens, re-enable the button when the thread ends.
            self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
            self.worker.start()

        def on_done(self, res):
            self.run_btn.setEnabled(True)
            iq = res["iq"]
            self.c_i.setData(res["t_ms"], iq.real)
            self.c_q.setData(res["t_ms"], iq.imag)
            self.c_m.setData(res["t_ms"], np.abs(iq))
            self.c_s.setData(res["f_khz"], res["mag"])
            if res.get("flodict"):           # console seq: draw its compiled diagram
                self._draw_flodict(res["flodict"])
            self.log(f"Received {len(iq)} samples.  {res['msgs']}")

        def on_failed(self, err):
            self.run_btn.setEnabled(True)
            self.log("ERROR: " + err)
            self.log("Tip: is marcos_server running on the board, and the IP correct?")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(MRI4ALL_QSS)

    # Pre-load the heavy marcos/matplotlib import in the background so the FIRST
    # real scan isn't delayed by it (~250 ms one-time cost moved off the click).
    import threading
    def _prewarm():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot  # noqa: F401  (the slow part)
        except Exception:
            pass
    threading.Thread(target=_prewarm, daemon=True).start()

    win = Main()
    if smoke:
        win.resize(1180, 720)
        shot = os.environ.get("RPCONSOLE_SHOT")
        if shot:
            # fill plots with a demo scan, then render both screens to PNGs
            demo_params = dict(ip="x", port=11111, freq_mhz=2.0, rf_len_us=10.0,
                               rf_amp=0.5, rx_dwell_us=3.125, n_samples=512, deadtime_us=30.0)
            win.on_done(run_experiment(demo_params, demo=True))
            win._draw_sequence(demo_params)
            win.stack.setCurrentIndex(1)
            for tab, tag in [(0, "exam"), (1, "seq")]:
                win.tabs.setCurrentIndex(tab); app.processEvents(); app.processEvents()
                win.grab().save(f"{shot}_{tag}.png")
            win.stack.setCurrentIndex(0); app.processEvents(); app.processEvents()
            win.grab().save(f"{shot}_home.png")
            # spin-echo sequence diagram
            win.stack.setCurrentIndex(1); win.seq_combo.setCurrentIndex(1)
            win.tabs.setCurrentIndex(1); app.processEvents(); app.processEvents()
            win.grab().save(f"{shot}_seq_se.png")
            # loopback acquisition (demo -> plateau while TX is on)
            win.seq_combo.setCurrentIndex(2)
            win.on_done(run_experiment(dict(demo_params, sequence="loopback", rf_amp=0.3), demo=True))
            win.tabs.setCurrentIndex(0); app.processEvents(); app.processEvents()
            win.grab().save(f"{shot}_loopback.png")
            print(f"screenshots saved: {shot}_home.png, {shot}_exam.png, {shot}_seq.png, {shot}_seq_se.png, {shot}_loopback.png")
        else:
            win.stack.setCurrentIndex(1); app.processEvents()
            print(f"smoketest OK: window built, {win.stack.count()} screens, "
                  f"title='{win.windowTitle()}'")
        return
    win.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------

def selftest():
    """Exercise sequence + DSP paths with no Qt / hardware."""
    p = dict(ip="x", port=11111, freq_mhz=2.0, rf_len_us=10.0, rf_amp=0.5,
             rx_dwell_us=3.125, n_samples=512, deadtime_us=30.0)
    fd, meta = build_fid_flodict(p["rf_len_us"], p["rf_amp"], p["rx_dwell_us"], p["n_samples"])
    assert set(fd) == {"tx0", "tx_gate", "rx0_en"}, fd.keys()
    res = run_experiment(p, demo=True)
    assert len(res["iq"]) == 512 and len(res["f_khz"]) == 512
    peak_khz = res["f_khz"][int(np.argmax(res["mag"]))]
    # sequence-diagram waveforms
    wf = sequence_waveforms(fd)
    assert set(wf) == {"tx0", "tx_gate", "rx0_en"}, wf.keys()
    assert wf["tx0"][1].max() == 0.5, "TX amplitude step wrong"
    assert wf["rx0_en"][1].max() == 1.0, "RX window step wrong"
    # spin-echo: two RF pulses, RX window centred on the echo at TE
    se_p = dict(p, sequence="se", te_ms=10.0)
    se_fd, se_meta = build_sequence(se_p)
    assert (se_fd["tx0"][1] != 0).sum() == 2, "SE should have two RF pulses"
    assert abs(se_meta["echo_us"] - (10 + 0.5 * p["rf_len_us"] + 10e3)) < 1, "echo not at TE"
    se_res = run_experiment(se_p, demo=True)
    assert len(se_res["iq"]) == 512
    # loopback: TX window overlaps the RX window; demo shows a strong central plateau
    lb_p = dict(p, sequence="loopback")
    lb_fd, lb_meta = build_sequence(lb_p)
    tx_s, tx_e = lb_meta["tx_window_us"]
    assert lb_meta["rx_start_us"] < tx_s and tx_e < lb_meta["rx_end_us"], "TX must sit inside RX window"
    lb_res = run_experiment(lb_p, demo=True)
    mid = np.abs(lb_res["iq"][256]); edge = np.abs(lb_res["iq"][5])
    assert mid > 10 * edge, "loopback center should be >> edges"
    print(f"selftest OK: FID acq={meta['acq_len_us']:.0f}us samples={len(res['iq'])} "
          f"peak~{peak_khz:.2f}kHz; seq_wf={sorted(wf)}; "
          f"SE echo@{se_meta['echo_us']/1e3:.1f}ms 2 pulses, samples={len(se_res['iq'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run headless DSP/sequence checks and exit")
    ap.add_argument("--smoketest", action="store_true",
                    help="build the GUI offscreen and exit (no event loop)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.smoketest:
        launch_gui(smoke=True)
    else:
        launch_gui()

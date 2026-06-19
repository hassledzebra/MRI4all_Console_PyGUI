"""
Console sequence engine — reuse the real mri4all/console sequence builders.

Instead of re-implementing pulse sequences, this bootstraps just enough of the
console's environment (NO PyQt5, NO registry auto-discovery, NO services/IPC) to
call the console's actual pypulseq builders, convert them to a marcos flodict via
flocra-pulseq, and run them on our marcos_server.

Pipeline (same as the console's run_pulseq):
    builder(inputs) -> .seq  ->  PSInterpreter.interpret() -> flodict, params
    -> Experiment(lo_freq, rx_t).add_flodict(flodict).run() -> rxd["rx0"]

The console repo must be present at ../marcos/console.
"""
import os
import sys
import types
import logging
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(cands, marker):
    """First candidate dir that contains `marker` (file or subdir); else cands[0]."""
    for c in cands:
        if os.path.exists(os.path.join(c, marker)):
            return c
    return cands[0]


# Works in the repo layout (console vendored as a sibling subfolder) and in the
# dev layout (console at ../marcos/console).
CONSOLE = _find([
    os.path.join(HERE, "mri4all_console"),                              # repo layout
    os.path.normpath(os.path.join(HERE, "..", "marcos", "console")),    # dev layout
], os.path.join("sequences", "common"))
MARCOS_CLIENT = _find([
    os.path.normpath(os.path.join(HERE, "..", "marcos", "marcos_client")),  # dev
    os.path.join(HERE, "marcos_client"),                                    # repo (vendored)
    os.path.join(CONSOLE, "external", "marcos_client"),                     # console-bundled
], "experiment.py")
ENGINE_DIR = os.path.join(HERE, "engine")

_booted = False


def available():
    """True if the console repo is present so the engine can be used."""
    return os.path.isdir(os.path.join(CONSOLE, "sequences", "common"))


def bootstrap():
    """Set up paths + neutralize the console's Linux/UI couplings (idempotent)."""
    global _booted
    if _booted:
        return
    if not available():
        raise RuntimeError(f"console repo not found at {CONSOLE}")
    os.makedirs(os.path.join(ENGINE_DIR, "config"), exist_ok=True)
    for p in (CONSOLE, os.path.join(CONSOLE, "external")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # point the console's config/logs at a local writable dir (not /opt/mri4all)
    import common.runtime as rt
    rt.base_path = ENGINE_DIR
    import common.logger as clog
    clog.get_logger = lambda *a, **k: logging.getLogger("rpc_engine")
    # no-op IPC stub (the console's real one needs PyQt5 + the UI service)
    ipc = types.ModuleType("common.ipc")

    class Communicator:
        ACQ = "acq"; RECON = "recon"; UI = "ui"

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return lambda *a, **k: None

    ipc.Communicator = Communicator
    sys.modules["common.ipc"] = ipc
    sys.modules["common.ipc.ipc"] = ipc
    # replace the 'sequences' package with a bare one to skip its registry
    # auto-discovery (which imports every sequence -> PyQt5/DICOM/etc.)
    seqpkg = types.ModuleType("sequences")
    seqpkg.__path__ = [os.path.join(CONSOLE, "sequences")]
    seqpkg.__package__ = "sequences"
    sys.modules["sequences"] = seqpkg
    _booted = True


def set_larmor(mhz):
    bootstrap()
    import external.seq.adjustments_acq.config as cfg
    cfg.LARMOR_FREQ = float(mhz)


# Console sequence builders that import cleanly (live in sequences/common/).
# 'gradients' flags sequences that need the GPA gradient board to be meaningful.
SEQUENCES = {
    "rf_se": dict(
        label="RF Spin Echo (no gradients)", module="make_rf_se", fn="pypulseq_rfse",
        gradients=False,
        defaults=dict(TE=10, TR=500, NSA=1, FA1=90, FA2=180,
                      ADC_samples=512, ADC_duration=6400)),
    "se_1D": dict(
        label="Spin Echo 1D", module="make_se_1D", fn="pypulseq_1dse",
        gradients=True,
        defaults=dict(TE=20, TR=500, NSA=1, FOV=20, Base_Resolution=64,
                      BW=32000, Gradient="x")),
    "se_2D": dict(
        label="Spin Echo 2D", module="make_se_2D", fn="pypulseq_se2D",
        gradients=True,
        defaults=dict(TE=20, TR=500, NSA=1, FOV=20, Base_Resolution=64,
                      BW=32000, Orientation="0", view_traj=False)),
    "tse_2D": dict(
        label="Turbo Spin Echo 2D", module="make_tse_2D", fn="pypulseq_tse2D",
        gradients=True, defaults=dict(TE=20, TR=500)),
    "tse_3D": dict(
        label="Turbo Spin Echo 3D", module="make_tse_3D", fn="pypulseq_tse3D",
        gradients=True,
        defaults=dict(TE=20, TR=500, NSA=1, FOV=20, Base_Resolution=32, BW=32000,
                      ETL=4, FA1=90, FA2=180, Ordering="0", Orientation="0",
                      Slices=4, dummy_shots=0)),
}


def list_sequences():
    return {k: v["label"] for k, v in SEQUENCES.items()}


def defaults_for(name):
    return dict(SEQUENCES[name]["defaults"])


def build_seq(name, params, outfile):
    """Run the console builder for `name` with `params`, writing a .seq file."""
    bootstrap()
    spec = SEQUENCES[name]
    import importlib
    mod = importlib.import_module(f"sequences.common.{spec['module']}")
    fn = getattr(mod, spec["fn"])
    inputs = dict(spec["defaults"])
    inputs.update(params or {})
    ok = fn(inputs=inputs, check_timing=True, output_file=outfile)
    if not ok:
        raise RuntimeError(f"{name}: builder reported failure (check parameters)")
    return outfile


def interpret(seqfile, rf_center_mhz, rf_max=None, grad_max=1e7, tx_warmup=100):
    """Convert a .seq file to a marcos flodict. Returns (flodict, param_dict)."""
    bootstrap()
    import external.seq.adjustments_acq.config as cfg
    from flocra_pulseq.interpreter import PSInterpreter
    psi = PSInterpreter(rf_center=rf_center_mhz * 1e6,
                        rf_amp_max=rf_max if rf_max else cfg.RF_MAX,
                        grad_max=grad_max, tx_warmup=tx_warmup)
    return psi.interpret(seqfile)


def run_seq(name, params, ip, rf_center_mhz, port=11111, timeout_s=30.0):
    """Build -> interpret -> run a console sequence on the board.

    Returns dict: {iq, rx_t_us, readouts, flodict, msgs}.
    """
    import numpy as np
    import socket as _socket
    bootstrap()
    set_larmor(rf_center_mhz)
    seqfile = os.path.join(tempfile.gettempdir(), f"rpc_{name}.seq")
    build_seq(name, params, seqfile)
    flo, pd = interpret(seqfile, rf_center_mhz)

    # Use the console's BUNDLED marcos_client (version-matched to its
    # flocra_pulseq) — our separately-cloned latest one has an incompatible
    # compiler ("Tried to set a buffer to two values at once").
    import common.config as cconfig

    class _Cfg:
        scanner_ip = ip
    try:
        cconfig.Configuration.load_from_file = classmethod(lambda cls, *a, **k: _Cfg())
    except Exception:
        pass
    import external.marcos_client.experiment as ex
    ex.ip_address = ip          # override the module global used at connect
    ex.port = int(port)
    old = _socket.getdefaulttimeout()
    _socket.setdefaulttimeout(float(timeout_s))
    expt = None
    # No gradient board attached → drop the (zero) gradient channels. This also
    # avoids a numpy-2 overflow in the console client's gpa-fhdo encoder.
    flo_run = {k: v for k, v in flo.items() if not k.startswith("grad")}
    try:
        expt = ex.Experiment(lo_freq=rf_center_mhz, rx_t=pd["rx_t"],
                             init_gpa=False, halt_and_reset=True)
        expt.add_flodict(flo_run)
        rxd, msgs = expt.run()
    except (_socket.timeout, OSError) as e:
        raise RuntimeError(f"No response from {ip}:{port} ({type(e).__name__}). "
                           f"marcos_server running and free?")
    finally:
        _socket.setdefaulttimeout(old)
        if expt is not None:
            try:
                expt._s.close()
            except Exception:
                pass
    iq = rxd.get("rx0", np.array([], dtype=np.complex128))
    return {"iq": iq, "rx_t_us": pd["rx_t"], "readouts": pd.get("readout_number"),
            "flodict": flo, "msgs": "\n".join(str(m) for m in msgs) if msgs else "OK"}


if __name__ == "__main__":
    # Smoke-check: build + interpret every registered sequence (no board needed).
    if not available():
        print("console repo not present — engine unavailable"); raise SystemExit(1)
    set_larmor(2.0)
    for nm in SEQUENCES:
        try:
            f = os.path.join(tempfile.gettempdir(), f"rpc_{nm}.seq")
            build_seq(nm, {}, f)
            flo, pd = interpret(f, 2.0)
            print(f"  {nm:8s} OK  -> channels={len(flo)} readouts={pd.get('readout_number')} "
                  f"rx_t={pd['rx_t']:.3f}us")
        except Exception as e:
            print(f"  {nm:8s} FAIL: {type(e).__name__}: {str(e)[:80]}")

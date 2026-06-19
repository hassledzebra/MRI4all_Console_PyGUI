#!/usr/bin/env python3
"""
TX → RX loopback test for the Red Pitaya / MaRCoS console.

Physically cable the Red Pitaya's RF OUTPUT to its RF INPUT, then run this to
confirm the transmit/receive chain works end-to-end. With TX cabled to RX, the
receiver samples the transmitted pulse directly: you should see the signal jump
well above the noise floor while TX is on.

  ⚠️  SAFETY: Do NOT connect TX straight to RX at high amplitude — the transmit
      level can overload (or damage) the RX ADC. Use an SMA ATTENUATOR (20–30 dB)
      in line, and/or start at a LOW RF amplitude (--amp 0.1) and increase slowly.
      If the received |signal| flat-tops (saturates), reduce amplitude / add
      attenuation.

Usage:
    python loopback_test.py                      # real board at rp-f0d431.local
    python loopback_test.py --ip 169.254.160.215 --amp 0.2
    python loopback_test.py --demo               # no hardware (shows expected PASS)
"""
import argparse
import numpy as np
import rp_console as rc


def loopback_test(ip="rp-f0d431.local", freq_mhz=2.0, rf_amp=0.2,
                  n_samples=512, rx_dwell_us=3.125, demo=False):
    """Run one TX→RX loopback acquisition and report PASS/FAIL.

    Returns the result dict from rp_console.run_experiment.
    """
    params = dict(ip=ip, port=11111, freq_mhz=freq_mhz, rf_len_us=10.0,
                  rf_amp=rf_amp, rx_dwell_us=rx_dwell_us, n_samples=n_samples,
                  deadtime_us=30.0, sequence="loopback", te_ms=10.0)
    res = rc.run_experiment(params, demo=demo)
    iq = res["iq"]
    n = len(iq)
    tx_window = np.abs(iq[int(0.3 * n):int(0.7 * n)]).mean()   # while TX is on
    noise = np.abs(np.r_[iq[:int(0.05 * n)], iq[-int(0.05 * n):]]).mean()  # edges
    ratio = tx_window / noise if noise else float("inf")
    peak = np.abs(iq).max()

    print(f"  samples         : {n}")
    print(f"  TX-window |sig| : {tx_window:.3e}")
    print(f"  noise   |sig|   : {noise:.3e}")
    print(f"  ratio           : {ratio:.1f}×")
    print(f"  peak |sig|      : {peak:.3e}")
    if ratio > 5:
        print("  RESULT: ✅ PASS — strong TX→RX coupling (loopback detected).")
    else:
        print("  RESULT: ⚪ no signal — is the cable connected? attenuator too "
              "high, or RF amplitude too low? (raise --amp a little).")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Red Pitaya TX→RX loopback test")
    ap.add_argument("--ip", default="rp-f0d431.local", help="board host/IP")
    ap.add_argument("--freq", type=float, default=2.0, help="center freq (MHz)")
    ap.add_argument("--amp", type=float, default=0.2, help="RF amplitude 0–1 (start low!)")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--demo", action="store_true", help="synthetic, no hardware")
    a = ap.parse_args()
    print(f"TX→RX loopback test  ({'DEMO' if a.demo else a.ip})  amp={a.amp}")
    loopback_test(ip=a.ip, freq_mhz=a.freq, rf_amp=a.amp, n_samples=a.samples, demo=a.demo)

"""
Image reconstruction for the Red Pitaya MaRCoS console.

Reuses the **original mri4all/console** reconstruction math: the centered,
ortho-normalized FFT utilities from `mri4all_console/recon/recon_utils/imaging.py`
(`centered_ifft2`, `centered_fft2`, n-dim `centered_ifft`/`centered_fft`). Those
functions are loaded directly from the vendored console tree, so this is the same
code the console uses — not a re-implementation. A small fallback with identical
math is provided if the console tree isn't present.

Includes a synthetic phantom + forward model so the full encode → reconstruct
pipeline can be demonstrated/tested without gradient hardware (real k-space needs
the GPA gradient board + RF coil + B0 magnet).

The console's full pipeline (B0 correction, k-space filtering, ISMRMRD, DICOM
export) lives under `mri4all_console/recon/` and `mri4all_console/services/recon/`.
"""
import os
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_console_imaging():
    """Load the console's recon_utils/imaging.py directly from the vendored tree
    (it only imports numpy, so no heavy package side effects)."""
    import seq_engine  # locates the console tree (repo or dev layout)
    path = os.path.join(seq_engine.CONSOLE, "recon", "recon_utils", "imaging.py")
    spec = importlib.util.spec_from_file_location("mri4all_imaging", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _img = _load_console_imaging()
    centered_fft2 = _img.centered_fft2
    centered_ifft2 = _img.centered_ifft2
    centered_fftn = _img.centered_fft
    centered_ifftn = _img.centered_ifft
    RECON_SOURCE = "mri4all_console/recon/recon_utils/imaging.py"
except Exception:                                   # identical math, standalone
    def centered_fft2(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

    def centered_ifft2(y):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))

    def centered_fftn(x):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm="ortho"))

    def centered_ifftn(y):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y), norm="ortho"))
    RECON_SOURCE = "builtin (console tree not found)"


def recon_2d(kspace):
    """Cartesian 2D reconstruction → magnitude image (console centered iFFT)."""
    return np.abs(centered_ifft2(np.asarray(kspace)))


def recon_3d(kspace):
    """Cartesian 3D reconstruction → magnitude volume."""
    return np.abs(centered_ifftn(np.asarray(kspace)))


def forward_2d(image):
    """Image → Cartesian k-space (console centered FFT)."""
    return centered_fft2(np.asarray(image))


def shepp_logan(n=64):
    """A small Shepp–Logan-like phantom (ellipses), no external deps."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    img = np.zeros((n, n), dtype=float)
    ellipses = [  # (intensity, a, b, x0, y0, angle_deg)
        (1.0, 0.69, 0.92, 0.0, 0.0, 0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
        (0.3, 0.21, 0.25, 0.0, 0.35, 0),
        (0.2, 0.046, 0.046, 0.0, 0.10, 0),
        (0.2, 0.046, 0.023, -0.08, -0.605, 0),
        (0.2, 0.023, 0.023, 0.0, -0.606, 0),
    ]
    for inten, a, b, x0, y0, ang in ellipses:
        t = np.deg2rad(ang)
        xr = (x - x0) * np.cos(t) + (y - y0) * np.sin(t)
        yr = -(x - x0) * np.sin(t) + (y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += inten
    return np.clip(img, 0, None)


def phantom_kspace(n=64, noise=0.0, seed=0):
    """Synthetic acquisition: phantom image → k-space (+ optional complex noise)."""
    k = forward_2d(shepp_logan(n))
    if noise:
        rng = np.random.default_rng(seed)
        scale = noise * np.max(np.abs(k))
        k = k + scale * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
    return k


if __name__ == "__main__":
    truth = shepp_logan(64)
    img = recon_2d(phantom_kspace(64, noise=0.005))
    img /= img.max()
    corr = np.corrcoef(truth.ravel(), img.ravel())[0, 1]
    print(f"recon selftest OK: source={RECON_SOURCE}; 64x64 phantom, "
          f"recon/truth corr={corr:.3f}")

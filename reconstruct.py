"""
Image reconstruction + post-processing + DICOM export for the Red Pitaya console.

Reuses the **original mri4all/console** code wherever possible:
  - centered FFT recon  : mri4all_console/recon/recon_utils/imaging.py
  - k-space apodization : mri4all_console/recon/kspaceFiltering/kspace_filtering.py
  - image denoising     : mri4all_console/recon/image_filters/denoise.py
  - DICOM export        : pydicom pattern from mri4all_console/recon/DICOM/

(Named `reconstruct` rather than `recon` to avoid colliding with the console's
own top-level `recon` package.)

A synthetic phantom + forward model lets the encode → reconstruct pipeline be
demonstrated without gradient hardware (real k-space needs the GPA gradient
board + RF coil + B0 magnet).
"""
import os
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_console_imaging():
    """Load the console's recon_utils/imaging.py directly from the vendored tree
    (only imports numpy, so no heavy package side effects)."""
    import seq_engine
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


def kspace_filter(kspace, kind="none"):
    """Apodize k-space to suppress Gibbs ringing. Reuses the console's
    `kFilter` (fermi / gaussian / sine_bell). kind='none' is a passthrough."""
    if not kind or kind == "none":
        return kspace
    import seq_engine
    seq_engine.bootstrap()
    from recon.kspaceFiltering.kspace_filtering import kFilter   # console code
    return kFilter(np.asarray(kspace), filter_type=kind, center_correction=False)


def denoise_image(image, sigma=0.0):
    """Denoise a magnitude image. Reuses the console's
    `image_filters.denoise.remove_gaussian_noise`. sigma<=0 is a passthrough."""
    if not sigma or sigma <= 0:
        return image
    import seq_engine
    seq_engine.bootstrap()
    import recon.image_filters.denoise as dn                     # console code
    return np.asarray(dn.remove_gaussian_noise(np.asarray(image), sigma=float(sigma)))


def export_dicom(image, params=None, outdir=None, patient=None):
    """Write a magnitude image as a DICOM MR file (pydicom pattern from the
    console's recon/DICOM). Returns the file path."""
    import datetime
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, MRImageStorage
    params = params or {}
    patient = patient or {}
    img = np.asarray(image, dtype=float)
    img = img - img.min()
    px = (img / (img.max() or 1.0) * 65535).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    now = datetime.datetime.now()
    ds.PatientName = patient.get("name", "PHANTOM^MRI4ALL")
    ds.PatientID = patient.get("mrn", "000000")
    ds.Modality = "MR"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesDescription = str(params.get("sequence", "scan"))
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = px.tobytes()

    outdir = outdir or os.path.join(HERE, "data")
    os.makedirs(outdir, exist_ok=True)
    seq = str(params.get("sequence", "scan")).replace(":", "_")
    path = os.path.join(outdir, f"{seq}_{now.strftime('%Y%m%d_%H%M%S')}.dcm")
    ds.save_as(path, write_like_original=False)
    return path


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
    k = phantom_kspace(64, noise=0.02)
    img = recon_2d(kspace_filter(k, "fermi"))
    img = denoise_image(img, sigma=0.6)
    img /= img.max()
    corr = np.corrcoef(truth.ravel(), img.ravel())[0, 1]
    print(f"reconstruct selftest OK: source={RECON_SOURCE}; phantom corr={corr:.3f}; "
          f"kspace_filter+denoise reused from console")

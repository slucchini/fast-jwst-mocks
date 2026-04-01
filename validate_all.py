"""
Validate cell-based 7.7 um emissivity against SKIRT F770W at multiple viewing angles.

Compares face-on (fo), zoom, ic5332 (inc=27), inc60, and edge-on (eo) instruments.
Produces a multi-panel figure with images and radial profiles for each.

Usage:
    python validate_all.py
"""

import sys
import os
import argparse

pts_path = os.environ.get("PTS_PATH")
if pts_path:
    sys.path.insert(0, pts_path)

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import pts.band as bnd
import astropy.units as u

from project import make_projection

# ── SKIRT instrument definitions ──────────────────────────────────────────
INSTRUMENTS = [
    {
        "name": "fo",
        "fits": "../I5_output/smuggle_fo_total.fits",
        "inc": 0, "az": 0,
        "fov_kpc": 40.0, "npix": 512, "distance_Mpc": 1.0,
        "zoom_kpc": 20.0,
        "label": "Face-on (40 kpc)",
    },
    {
        "name": "zoom",
        "fits": "../I5_output/smuggle_zoom_total.fits",
        "inc": 0, "az": 0,
        "fov_kpc": 10.0, "npix": 1024, "distance_Mpc": 1.0,
        "zoom_kpc": 10.0,
        "label": "Zoom (10 kpc)",
    },
    {
        "name": "ic5332",
        "fits": "../I5_output/smuggle_ic5332_total.fits",
        "inc": 27, "az": 0,
        "fov_kpc": 10.0, "npix": 1024, "distance_Mpc": 8.84,
        "zoom_kpc": 10.0,
        "label": "IC 5332 (inc=27°)",
    },
    {
        "name": "inc60",
        "fits": "../I5_output/smuggle_inc60_total.fits",
        "inc": 60, "az": 0,
        "fov_kpc": 40.0, "npix": 512, "distance_Mpc": 1.0,
        "zoom_kpc": 20.0,
        "label": "inc=60°",
    },
    {
        "name": "eo",
        "fits": "../I5_output/smuggle_eo_total.fits",
        "inc": 90, "az": 0,
        "fov_kpc": 40.0, "npix": 512, "distance_Mpc": 1.0,
        "zoom_kpc": 20.0,
        "label": "Edge-on",
    },
]

def load_skirt_f770w(fits_path):
    """Load SKIRT datacube and convolve with MIRI F770W filter."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data     # (nwav, ny, nx) in MJy/sr
        nwav = hdul[0].header["NAXIS3"]
        npix = hdul[0].header["NAXIS1"]

    # Reconstruct wavelength grid (nested: 90 base + 100 sub in 5.5-14 um)
    if nwav == 90:
        wavelengths = np.logspace(np.log10(0.09), np.log10(100), 90)
    else:
        base = np.logspace(np.log10(0.09), np.log10(100), 90)
        sub  = np.logspace(np.log10(5.5), np.log10(14), 100)
        base_outside = base[(base < sub.min()) | (base > sub.max())]
        wavelengths = np.sort(np.concatenate([base_outside, sub]))
        if len(wavelengths) != nwav:
            wavelengths = np.logspace(np.log10(0.09), np.log10(100), nwav)

    band = bnd.builtinBand("JWST_MIRI_F770W")
    image_f770w = band.convolve(
        wavelengths * u.micron,
        np.transpose(data, [1, 2, 0]) * u.MJy / u.sr,
    ).value

    return image_f770w, npix


def radial_profile(img, npix, fov_kpc, nbins=40):
    """Compute azimuthally averaged radial profile (mean over all pixels)."""
    iy, ix = np.mgrid[:npix, :npix]
    r_kpc = np.sqrt((ix - npix/2)**2 + (iy - npix/2)**2) * fov_kpc / npix

    rbins = np.linspace(0, fov_kpc / 2, nbins + 1)
    rcenters = 0.5 * (rbins[:-1] + rbins[1:])

    prof = np.zeros(nbins)
    for i in range(nbins):
        mask = (r_kpc >= rbins[i]) & (r_kpc < rbins[i+1])
        prof[i] = np.nanmean(img[mask]) if np.any(mask) else np.nan

    return rcenters, prof


def main():
    parser = argparse.ArgumentParser(
        description="Validate cell emissivity against SKIRT at multiple viewing angles")
    parser.add_argument("--emissivity", default="emissivity_snap190.h5",
                        help="Emissivity HDF5 file from compute_emissivity.py")
    parser.add_argument("--skirt-dir", default="../I5_output",
                        help="Directory containing SKIRT FITS files (default: ../I5_output)")
    parser.add_argument("-o", "--output", default="validation_all_angles.png",
                        help="Output figure path")
    args = parser.parse_args()

    # Update FITS paths relative to --skirt-dir
    for inst in INSTRUMENTS:
        basename = os.path.basename(inst["fits"])
        inst["fits"] = os.path.join(args.skirt_dir, basename)

    n_inst = len(INSTRUMENTS)

    fig, axes = plt.subplots(3, n_inst, figsize=(4 * n_inst, 12))

    all_alphas = {}

    for col, inst in enumerate(INSTRUMENTS):
        print(f"\n{'='*50}")
        print(f"Processing {inst['name']} ({inst['label']})...")

        # Load SKIRT
        skirt_img, npix_skirt = load_skirt_f770w(inst["fits"])
        print(f"  SKIRT: {npix_skirt}x{npix_skirt}, FOV={inst['fov_kpc']} kpc")

        # Cell projection
        _, cell_img = make_projection(
            args.emissivity,
            inc_deg=inst["inc"], az_deg=inst["az"],
            fov_kpc=inst["fov_kpc"],
            npix=inst["npix"],
            distance_Mpc=inst["distance_Mpc"],
        )
        cell_img = cell_img[:, ::-1]

        # Radial profiles
        rcenters, skirt_prof = radial_profile(skirt_img, npix_skirt, inst["fov_kpc"])
        _, cell_prof = radial_profile(cell_img, npix_skirt, inst["fov_kpc"])

        # Calibration factor (inner half of zoom region)
        r_inner = inst["zoom_kpc"] / 2
        inner = ((rcenters < r_inner) & np.isfinite(skirt_prof) &
                 np.isfinite(cell_prof) & (cell_prof > 0))
        alpha = np.nanmedian(skirt_prof[inner] / cell_prof[inner]) if np.any(inner) else 1.0
        all_alphas[inst["name"]] = alpha
        print(f"  alpha = {alpha:.3f}")

        # Shared color scale
        half = inst["fov_kpc"] / 2
        extent = [-half, half, -half, half]
        zh = inst["zoom_kpc"] / 2
        vmin, vmax = 1e-2, 60

        # Row 1: SKIRT
        ax = axes[0, col]
        ax.imshow(skirt_img, origin="lower", extent=extent,
                  norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
        ax.set_xlim(-zh, zh)
        ax.set_ylim(-zh, zh)
        ax.set_title(f"SKIRT — {inst['label']}", fontsize=10)
        if col == 0:
            ax.set_ylabel("y (kpc)")

        # Row 2: Cell
        ax = axes[1, col]
        ax.imshow(cell_img, origin="lower", extent=extent,
                  norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
        ax.set_xlim(-zh, zh)
        ax.set_ylim(-zh, zh)
        ax.set_title(f"Cell — α={alpha:.2f}", fontsize=10)
        if col == 0:
            ax.set_ylabel("y (kpc)")

        # Row 3: Radial profiles
        ax = axes[2, col]
        ax.semilogy(rcenters, skirt_prof, "k-", lw=2, label="SKIRT")
        ax.semilogy(rcenters, cell_prof, "r--", lw=1.5, label="Cell (raw)")
        ax.semilogy(rcenters, cell_prof * alpha, "r-", lw=2,
                    label=f"Cell (×{alpha:.2f})")
        ax.set_xlabel("r (kpc)")
        if col == 0:
            ax.set_ylabel("SB (MJy/sr)")
        ax.legend(fontsize=7)
        ax.set_xlim(0, zh)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n{'='*50}")
    print(f"Saved {args.output}")
    print(f"\nCalibration factors:")
    for name, alpha in all_alphas.items():
        print(f"  {name:10s}: α = {alpha:.3f}")

    vals = list(all_alphas.values())
    print(f"\n  Median α = {np.median(vals):.3f}")
    print(f"  Spread   = {np.max(vals)/np.min(vals):.2f}x")


if __name__ == "__main__":
    main()

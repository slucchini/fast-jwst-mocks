"""
Validate cell-based 7.7 um emissivity against SKIRT F770W face-on output.

Produces a 3-panel figure:
  1) SKIRT F770W (convolved from datacube)
  2) Cell-based projection (face-on)
  3) Radial profile comparison + calibration factor

Usage:
    python validate.py
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

    # Convolve with F770W
    band = bnd.builtinBand("JWST_MIRI_F770W")
    image_f770w = band.convolve(
        wavelengths * u.micron,
        np.transpose(data, [1, 2, 0]) * u.MJy / u.sr,
    ).value  # (npix, npix) in MJy/sr

    return image_f770w, npix


def main():
    parser = argparse.ArgumentParser(
        description="Validate cell emissivity against SKIRT F770W face-on output")
    parser.add_argument("--skirt", default="../I5_output/smuggle_zoom_total.fits",
                        help="SKIRT datacube FITS file")
    parser.add_argument("--emissivity", default="emissivity_snap190.h5",
                        help="Emissivity HDF5 file from compute_emissivity.py")
    parser.add_argument("-o", "--output", default="validation_comparison.png",
                        help="Output figure path")
    parser.add_argument("--fov", type=float, default=10.0,
                        help="Field of view in kpc (default: 10)")
    parser.add_argument("--distance", type=float, default=1.0,
                        help="Distance in Mpc (default: 1.0)")
    args = parser.parse_args()

    print("Loading SKIRT F770W...")
    skirt_img, npix_skirt = load_skirt_f770w(args.skirt)
    print(f"  SKIRT image: {npix_skirt}x{npix_skirt}, FOV={args.fov} kpc")

    print("Making cell-based face-on projection...")
    _, cell_img = make_projection(
        args.emissivity,
        inc_deg=0, az_deg=0,
        fov_kpc=args.fov,
        npix=npix_skirt,
        distance_Mpc=args.distance,
    )
    print(f"  Cell image: {cell_img.shape}")
    cell_img = cell_img[:,::-1]
    # cell_img[cell_img == 0] = np.min(cell_img[cell_img > 0])

    # ── Radial profiles ────────────────────────────────────────────────
    iy, ix = np.mgrid[:npix_skirt, :npix_skirt]
    r_kpc = np.sqrt((ix - npix_skirt/2)**2 + (iy - npix_skirt/2)**2) * args.fov / npix_skirt

    rbins = np.linspace(0, args.fov / 2, 40)
    rcenters = 0.5 * (rbins[:-1] + rbins[1:])

    skirt_prof = np.zeros(len(rbins) - 1)
    cell_prof  = np.zeros(len(rbins) - 1)

    for i in range(len(rbins) - 1):
        mask = (r_kpc >= rbins[i]) & (r_kpc < rbins[i+1])
        skirt_prof[i] = np.nanmean(skirt_img[mask]) if np.any(mask) else np.nan
        cell_prof[i] = np.nanmean(cell_img[mask]) if np.any(mask) else np.nan

    # Fit calibration factor using inner region (r < 5 kpc)
    inner = (rcenters < 5) & np.isfinite(skirt_prof) & np.isfinite(cell_prof) & (cell_prof > 0)
    if np.any(inner):
        alpha = np.nanmedian(skirt_prof[inner] / cell_prof[inner])
    else:
        alpha = 1.0

    print(f"\nCalibration factor alpha = {alpha:.3f}")
    print(f"  (multiply C_77 by {alpha:.3f} to match SKIRT)")

    # Read current C_77
    with h5py.File(args.emissivity, "r") as f:
        C_77_used = f["meta"].attrs["C_77"]
    C_77_calibrated = C_77_used * alpha
    print(f"  C_77 used:       {C_77_used:.3e}")
    print(f"  C_77 calibrated: {C_77_calibrated:.3e}")

    # ── 3-panel figure ─────────────────────────────────────────────────
    # Zoom to inner 10 kpc for comparison
    zoom_kpc = 10.0
    half = args.fov / 2
    extent = [-half, half, -half, half]

    # Shared color scale from SKIRT
    skirt_pos = skirt_img[skirt_img > 0]
    # vmin = np.percentile(skirt_pos, 5) if len(skirt_pos) else 0.01
    # vmax = np.percentile(skirt_pos, 99.5) if len(skirt_pos) else 10.0
    vmin, vmax = 1e-2, 60

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), width_ratios=[1,1,0.2,1])
    plt.subplots_adjust(wspace=0.02)
    axes[2].remove()

    # Panel 1: SKIRT
    ax = axes[0]
    skirt_img_fill = np.copy(skirt_img)
    skirt_img_fill[skirt_img==0] = np.min(skirt_img[skirt_img>0])
    im = ax.imshow(skirt_img_fill, origin="lower", extent=extent,
                norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
    ax.set_xlim(-zoom_kpc/2, zoom_kpc/2)
    ax.set_ylim(-zoom_kpc/2, zoom_kpc/2)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.text(0.02,0.98,"SKIRT F770W",va='top',transform=ax.transAxes,c='w',fontsize=14)

    # Panel 2: Cell-based (scaled by alpha)
    ax = axes[1]
    cell_img_fill = np.copy(cell_img * alpha)
    cell_img_fill[cell_img==0] = np.min(cell_img[cell_img>0])
    ax.imshow(cell_img_fill, origin="lower", extent=extent,
            norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
    ax.set_xlim(-zoom_kpc/2, zoom_kpc/2)
    ax.set_ylim(-zoom_kpc/2, zoom_kpc/2)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("")
    ax.text(0.98,0.98,f"Cell emissivity (×{alpha:.2f})",va='top',ha='right',transform=ax.transAxes,c='w',fontsize=14)
    ax.set_yticklabels([])

    fig.colorbar(im, ax=axes[:2], location='top', label="MJy/sr", fraction=0.03, pad=0.05, aspect=50)

    # Panel 3: Radial profiles
    ax = axes[-1]
    ax.semilogy(rcenters, skirt_prof, "k-", lw=2, label="SKIRT")
    ax.semilogy(rcenters, cell_prof, "r--", lw=1.5, label="Cell (raw)")
    ax.semilogy(rcenters, cell_prof * alpha, "r-", lw=2, label=f"Cell (×{alpha:.2f})")
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("Surface Brightness (MJy/sr)")
    ax.set_title("Radial Profiles")
    ax.legend(fontsize=9)
    ax.set_xlim(0, zoom_kpc / 2)

    box1 = axes[1].get_position()
    box2 = ax.get_position()
    ax.set_position([box2.x0,box1.y0,box2.width,box1.height])

    # fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
Project per-cell 7.7 um emissivities onto 2D images from arbitrary viewing angles.

Usage:
    python project.py                          # face-on (histogram)
    python project.py --inc 27 --az 0          # IC 5332 inclination
    python project.py --inc 60 --az 45         # arbitrary view
    python project.py --vortrace               # Voronoi projection (requires vortrace)
"""

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

# ── Constants ──────────────────────────────────────────────────────────────
kpc_cm  = 3.0857e21
Mpc_cm  = 3.0857e24
Jy_cgs  = 1e-23       # 1 Jy = 1e-23 erg/s/cm^2/Hz
MJy_cgs = 1e6 * Jy_cgs

# F770W effective wavelength and bandwidth
LAMBDA_77   = 7.7e-4      # cm
DELTA_NU_77 = 1.1e13      # Hz (F770W FWHM ~1.7 um → Δν ≈ c Δλ/λ² ≈ 8.6e12, use ~1.1e13 for full passband)


def rotate_positions(pos, inc_deg, az_deg):
    """Rotate positions by azimuth (around z) then inclination (around x).

    Returns rotated positions; projection is along new z-axis.
    """
    inc = np.radians(inc_deg)
    az  = np.radians(az_deg)

    ci, si = np.cos(inc), np.sin(inc)
    ca, sa = np.cos(az),  np.sin(az)

    # Rz (azimuth)
    x1 = pos[:, 0] * ca - pos[:, 1] * sa
    y1 = pos[:, 0] * sa + pos[:, 1] * ca
    z1 = pos[:, 2]

    # Rx (inclination)
    x2 = x1
    y2 = y1 * ci - z1 * si
    # z2 = y1 * si + z1 * ci  # line-of-sight, not needed

    return np.column_stack([x2, y2])


def make_projection(h5_path, inc_deg=0, az_deg=0,
                    fov_kpc=10.0, npix=512, distance_Mpc=8.84):
    """Project cell emissivities into a 2D surface brightness image.

    Parameters
    ----------
    h5_path : str
        Path to emissivity HDF5 file from compute_emissivity.py
    inc_deg, az_deg : float
        Viewing angles in degrees (0,0 = face-on)
    fov_kpc : float
        Field of view (side length) in kpc
    npix : int
        Number of pixels per side
    distance_Mpc : float
        Distance to galaxy in Mpc (for converting to MJy/sr)

    Returns
    -------
    image : (npix, npix) array
        Surface brightness in erg/s/kpc^2 (luminosity surface density)
    image_MJy_sr : (npix, npix) array
        Surface brightness in MJy/sr (observer units, monochromatic at 7.7 um)
    """
    with h5py.File(h5_path, "r") as f:
        pos  = f["gas/pos"][:]     # (N, 3) kpc
        j_77 = f["gas/j_77"][:]    # (N,) erg/s

    # Rotate and project
    pos_2d = rotate_positions(pos, inc_deg, az_deg)

    half = fov_kpc / 2.0
    bins = np.linspace(-half, half, npix + 1)

    # Weighted 2D histogram: sum j_77 per pixel → erg/s per pixel
    image, _, _ = np.histogram2d(
        pos_2d[:, 0], pos_2d[:, 1],
        bins=[bins, bins],
        weights=j_77,
    )
    image = image.T  # (y, x) convention for imshow

    # Pixel area in kpc^2
    pixel_kpc = fov_kpc / npix
    pixel_kpc2 = pixel_kpc**2

    # Surface luminosity density: erg/s/kpc^2
    image_lum = image / pixel_kpc2

    # Convert to MJy/sr (observer frame, monochromatic at 7.7 um)
    # Flux density = L_ν / (4π D²), where L_ν = L / Δν
    # Surface brightness = flux_density / pixel_solid_angle
    # pixel_solid_angle = (pixel_kpc * kpc_cm / (D_Mpc * Mpc_cm))^2 sr
    D_cm = distance_Mpc * Mpc_cm
    pixel_sr = (pixel_kpc * kpc_cm / D_cm)**2

    # L per pixel → F_ν per pixel = L / (4π D² Δν)  [erg/s/cm²/Hz]
    # Surface brightness = F_ν / pixel_sr  [erg/s/cm²/Hz/sr]
    image_MJy_sr = image / (4.0 * np.pi * D_cm**2 * DELTA_NU_77) / pixel_sr / MJy_cgs

    return image_lum, image_MJy_sr


def make_projection_vortrace(h5_path, inc_deg=0, az_deg=0,
                             fov_kpc=10.0, npix=512, distance_Mpc=8.84):
    """Project cell emissivities using vortrace Voronoi line integration.

    Unlike the histogram method, this traces rays through the Voronoi mesh
    and integrates emissivity density along each line of sight, properly
    handling the cell geometry.

    Parameters
    ----------
    h5_path : str
        Path to emissivity HDF5 file from compute_emissivity.py
    inc_deg, az_deg : float
        Viewing angles in degrees (0,0 = face-on)
    fov_kpc : float
        Field of view (side length) in kpc
    npix : int
        Number of pixels per side
    distance_Mpc : float
        Distance to galaxy in Mpc (for converting to MJy/sr)

    Returns
    -------
    image_lum : (npix, npix) array
        Surface brightness in erg/s/kpc^2
    image_MJy_sr : (npix, npix) array
        Surface brightness in MJy/sr
    """
    import vortrace as vt

    # Load emissivity data and cell volumes
    with h5py.File(h5_path, "r") as f:
        pos     = f["gas/pos"][:]      # (N, 3) kpc, centered on stellar COM
        j_77    = f["gas/j_77"][:]     # (N,) erg/s
        vol_kpc3 = f["gas/volume"][:]  # (N,) kpc^3

    # Emissivity density: erg/s/kpc^3
    j_density = j_77 / vol_kpc3

    # Set up vortrace projection cloud
    half = fov_kpc / 2.0
    depth = half  # kpc, integration depth (equal to fov)
    # bbox = [pos[:, 0].min(), pos[:, 0].max(),
    #         pos[:, 1].min(), pos[:, 1].max(),
    #         pos[:, 2].min(), pos[:, 2].max()]
    bbox = [-fov_kpc, fov_kpc,
            -fov_kpc, fov_kpc,
            -fov_kpc, fov_kpc]

    print("Building projection cloud...")
    stime = time.time()
    pc = vt.ProjectionCloud(pos, j_density, boundbox=bbox, vol=vol_kpc3)
    print("done ({:.2f} s)".format(time.time() - stime))

    # Grid extent and integration bounds (centered on origin)
    extent = [-half, half]
    bounds = [-depth, depth]
    center = [0.0, 0.0, 0.0]

    # vortrace uses Tait-Bryan angles: yaw = azimuth, pitch = inclination
    yaw   = np.radians(az_deg)
    pitch = np.radians(inc_deg)

    print("Making grid projection...")
    stime = time.time()
    image = pc.grid_projection(extent, npix, bounds, center,
                               yaw=yaw, pitch=pitch,
                               reduction='integrate')
    print("done ({:.2f} s)".format(time.time() - stime))

    # image is in erg/s/kpc^2 (line integral of erg/s/kpc^3 over kpc)
    image_lum = image.T  # transpose to (y, x) for imshow

    # Convert to MJy/sr
    D_cm = distance_Mpc * Mpc_cm
    pixel_kpc = fov_kpc / npix
    pixel_sr = (pixel_kpc * kpc_cm / D_cm)**2
    image_MJy_sr = image_lum * pixel_kpc**2 / (
        4.0 * np.pi * D_cm**2 * DELTA_NU_77) / pixel_sr / MJy_cgs

    return image_lum, image_MJy_sr


def plot_projection(image_MJy_sr, fov_kpc=10.0, inc_deg=0, az_deg=0,
                    vmin=None, vmax=None, out_path=None):
    """Plot a single projection image."""
    half = fov_kpc / 2.0
    extent = [-half, half, -half, half]

    # Auto scale
    pos_vals = image_MJy_sr[image_MJy_sr > 0]
    if vmin is None:
        vmin = np.percentile(pos_vals, 1) if len(pos_vals) > 0 else 1e-3
    if vmax is None:
        vmax = np.percentile(pos_vals, 99.5) if len(pos_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        image_MJy_sr, origin="lower", extent=extent,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno",
    )
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_title(f"7.7 μm cell emissivity — inc={inc_deg}°, az={az_deg}°")
    cb = fig.colorbar(im, ax=ax, label="MJy/sr")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Project 7.7 um emissivity map")
    parser.add_argument("--h5", default="emissivity_snap190.h5",
                        help="Input HDF5 file")
    parser.add_argument("--inc", type=float, default=0, help="Inclination (deg)")
    parser.add_argument("--az", type=float, default=0, help="Azimuth (deg)")
    parser.add_argument("--fov", type=float, default=10.0, help="FOV in kpc")
    parser.add_argument("--npix", type=int, default=512, help="Pixels per side")
    parser.add_argument("--dist", type=float, default=8.84, help="Distance in Mpc")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("--vortrace", action="store_true",
                        help="Use vortrace Voronoi projection (requires vortrace)")
    args = parser.parse_args()

    if args.vortrace:
        _, image_MJy_sr = make_projection_vortrace(
            args.h5, args.inc, args.az,
            args.fov, args.npix, args.dist)
    else:
        _, image_MJy_sr = make_projection(
            args.h5, args.inc, args.az, args.fov, args.npix, args.dist)

    out = args.output or f"proj_inc{args.inc:.0f}_az{args.az:.0f}.png"
    plot_projection(image_MJy_sr, args.fov, args.inc, args.az, out_path=out)


if __name__ == "__main__":
    main()

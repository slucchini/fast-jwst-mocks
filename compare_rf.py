"""
Compare geometric U-field estimate against SKIRT RadiationFieldProbe output.

SKIRT provides J(λ) [W/m²/Hz/sr] on a 2D planar cut (xy midplane).
We integrate over FUV (912–3000 Å), convert to energy density u, and
express in Mathis (1983) ISRF units for comparison with our cell-based U.
"""

import numpy as np
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Paths ─────────────────────────────────────────────────────────────────
SKIRT_RF = "../I5_output/smuggle_rf_J_xy.fits"
EMISSIVITY_FILE = "emissivity_snap190.h5"

# ── Constants ─────────────────────────────────────────────────────────────
c_cgs = 2.998e10          # cm/s
c_si  = 2.998e8           # m/s
u_ISRF = 8.64e-13         # erg/cm³, Mathis (1983) FUV energy density
kpc_cm = 3.0857e21
pc_cm  = 3.0857e18

# FUV band limits
FUV_WL_MIN = 0.0912  # μm (912 Å)
FUV_WL_MAX = 0.300   # μm (3000 Å)


def load_skirt_rf(path):
    """Load SKIRT radiation field probe and compute FUV energy density in Mathis units."""
    with fits.open(path) as hdul:
        J_cube = hdul[0].data       # (N_wl, Ny, Nx) in W/m²/Hz/sr
        header = hdul[0].header
        wl_um = hdul[1].data['GRID_POINTS']  # wavelengths in μm

    # Spatial grid info
    nx = header['NAXIS1']
    dx_pc = header['CDELT1']   # pc/pixel
    fov_kpc = nx * dx_pc / 1e3

    # Select FUV wavelengths
    fuv = (wl_um >= FUV_WL_MIN) & (wl_um <= FUV_WL_MAX)
    wl_fuv = wl_um[fuv]
    J_fuv = J_cube[fuv]  # (N_fuv, Ny, Nx) W/m²/Hz/sr

    # Convert wavelength to frequency: ν = c / λ
    freq_hz = c_si / (wl_fuv * 1e-6)  # Hz

    # Integrate J_ν over FUV frequency to get mean intensity J [W/m²/sr]
    # J_ν is in W/m²/Hz/sr; integrate over ν using trapezoidal rule
    # Frequencies decrease as wavelengths increase, so sort ascending
    sort_idx = np.argsort(freq_hz)
    freq_sorted = freq_hz[sort_idx]
    J_sorted = J_fuv[sort_idx]  # (N_fuv, Ny, Nx)

    # Integrate: J = ∫ J_ν dν  [W/m²/sr]
    J_integrated = np.trapz(J_sorted, freq_sorted, axis=0)  # (Ny, Nx)

    # Energy density: u = 4π J / c  [J/m³] → convert to erg/cm³
    # 1 J/m³ = 10 erg/cm³ (since 1 J = 1e7 erg, 1 m³ = 1e6 cm³)
    u_fuv_cgs = 4.0 * np.pi * J_integrated / c_si * 10.0  # erg/cm³

    # Express in Mathis units
    U_skirt = u_fuv_cgs / u_ISRF

    print(f"SKIRT RF probe: {nx}x{nx} grid, {dx_pc:.1f} pc/pix, FOV={fov_kpc:.1f} kpc")
    print(f"FUV wavelengths ({fuv.sum()}): {wl_fuv} μm")
    print(f"U_SKIRT: min={U_skirt.min():.3e}, max={U_skirt.max():.2f}, "
          f"median(>0)={np.median(U_skirt[U_skirt>0]):.3f}")

    return U_skirt, dx_pc, fov_kpc


def load_cell_U(path):
    """Load cell-based U-field and positions."""
    with h5py.File(path, "r") as f:
        pos = f["gas/pos"][:]       # (N, 3) kpc
        U = f["gas/U_field"][:]     # (N,) Mathis units
        M_dust = f["gas/M_dust"][:]
    return pos, U, M_dust


def project_U_to_grid(pos, U, M_dust, fov_kpc, npix, dz_kpc=1.0):
    """Project cell U-field onto 2D grid (xy) using dust-mass-weighted average.

    Select cells within |z| < dz_kpc of the midplane (matching the SKIRT
    planar cut which samples at z=0 but represents a density-weighted
    average through the grid cell).
    """
    half = fov_kpc / 2.0
    dx = fov_kpc / npix

    # Select cells in FOV and near midplane
    mask = ((np.abs(pos[:, 0]) < half) &
            (np.abs(pos[:, 1]) < half) &
            (np.abs(pos[:, 2]) < dz_kpc))
    print(f"Cells in FOV and |z|<{dz_kpc} kpc: {mask.sum()} / {len(pos)}")

    px = pos[mask, 0]
    py = pos[mask, 1]
    u_sel = U[mask]
    md_sel = M_dust[mask]

    # Pixel indices
    ix = ((px + half) / dx).astype(int)
    iy = ((py + half) / dx).astype(int)

    # Clip to grid
    ix = np.clip(ix, 0, npix - 1)
    iy = np.clip(iy, 0, npix - 1)

    # Dust-mass-weighted average U per pixel
    sum_mU = np.zeros((npix, npix), dtype=np.float64)
    sum_m  = np.zeros((npix, npix), dtype=np.float64)
    np.add.at(sum_mU, (iy, ix), md_sel * u_sel)
    np.add.at(sum_m,  (iy, ix), md_sel)

    U_proj = np.where(sum_m > 0, sum_mU / sum_m, 0.0)
    return U_proj


def main():
    # Load SKIRT radiation field
    U_skirt, dx_pc, fov_kpc = load_skirt_rf(SKIRT_RF)
    npix = U_skirt.shape[0]

    # Load cell-based estimate
    pos, U_cell, M_dust = load_cell_U(EMISSIVITY_FILE)

    # Project cell U onto same grid
    # Try a few midplane thickness values
    dz_kpc = 0.5  # ±500 pc slice
    U_proj = project_U_to_grid(pos, U_cell, M_dust, fov_kpc, npix, dz_kpc=dz_kpc)
    U_proj = (U_proj.T)[::-1,::-1]

    print(f"\nProjected cell U: min={U_proj.min():.3e}, max={U_proj.max():.2f}, "
          f"median(>0)={np.median(U_proj[U_proj>0]):.3f}")

    # ── Comparison plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Common normalization
    vmin, vmax = 1e-3, 30.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    extent = [-fov_kpc/2, fov_kpc/2, -fov_kpc/2, fov_kpc/2]

    # Panel 1: SKIRT U-field
    ax = axes[0, 0]
    im = ax.imshow(U_skirt, origin='lower', extent=extent, norm=norm, cmap='inferno')
    ax.set_title("SKIRT Radiation Field (FUV)")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax, label="U [Mathis]")

    # Panel 2: Cell-based U-field
    ax = axes[0, 1]
    im = ax.imshow(U_proj, origin='lower', extent=extent, norm=norm, cmap='inferno')
    ax.set_title(f"Geometric U-field (|z|<{dz_kpc} kpc)")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax, label="U [Mathis]")

    # Panel 3: Ratio map
    ax = axes[0, 2]
    valid = (U_skirt > vmin) & (U_proj > vmin)
    ratio = np.where(valid, U_proj / U_skirt, np.nan)
    im = ax.imshow(ratio, origin='lower', extent=extent,
                   norm=LogNorm(vmin=0.1, vmax=10), cmap='RdBu_r')
    ax.set_title("Ratio: Geometric / SKIRT")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax, label="U_cell / U_SKIRT")

    # Panel 4: Pixel-by-pixel scatter
    ax = axes[1, 0]
    mask_both = (U_skirt.ravel() > 1e-3) & (U_proj.ravel() > 1e-3)
    s_vals = U_skirt.ravel()[mask_both]
    c_vals = U_proj.ravel()[mask_both]
    ax.hexbin(s_vals, c_vals, gridsize=100, bins='log',
              xscale='log', yscale='log', cmap='viridis', mincnt=1)
    lims = [1e-3, 100]
    ax.plot(lims, lims, 'r--', lw=1.5, label='1:1')
    # Fit log-space offset
    log_ratio = np.log10(c_vals / s_vals)
    med_ratio = 10**np.median(log_ratio)
    ax.plot(lims, [lims[0]*med_ratio, lims[1]*med_ratio], 'r-', lw=1.5,
            label=f'median ratio = {med_ratio:.2f}')
    ax.set_xlabel("U_SKIRT [Mathis]")
    ax.set_ylabel("U_geometric [Mathis]")
    ax.set_title("Pixel-by-pixel comparison")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel 5: Radial profiles
    ax = axes[1, 1]
    # Build radial coordinate for each pixel
    y_grid, x_grid = np.mgrid[-fov_kpc/2:fov_kpc/2:npix*1j,
                               -fov_kpc/2:fov_kpc/2:npix*1j]
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    r_bins = np.linspace(0, 15, 60)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    prof_skirt = np.zeros(len(r_mid))
    prof_cell  = np.zeros(len(r_mid))
    for i in range(len(r_mid)):
        ring = (r_grid >= r_bins[i]) & (r_grid < r_bins[i+1])
        if ring.sum() > 0:
            prof_skirt[i] = np.median(U_skirt[ring])
            prof_cell[i]  = np.median(U_proj[ring])

    ax.semilogy(r_mid, prof_skirt, 'b-', lw=2, label='SKIRT')
    ax.semilogy(r_mid, prof_cell, 'r--', lw=2, label='Geometric')
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("U [Mathis]")
    ax.set_title("Radial profile (median)")
    ax.legend()
    ax.set_xlim(0, 15)
    ax.set_ylim(1e-3, 50)

    # Panel 6: Histogram of ratio
    ax = axes[1, 2]
    log_ratio_all = np.log10(c_vals / s_vals)
    ax.hist(log_ratio_all, bins=100, range=(-2, 2), density=True, alpha=0.7)
    ax.axvline(np.median(log_ratio_all), color='r', ls='--',
               label=f'median = {np.median(log_ratio_all):.2f} dex')
    ax.axvline(0, color='k', ls='-', alpha=0.3)
    std = np.std(log_ratio_all)
    ax.set_xlabel("log₁₀(U_geometric / U_SKIRT)")
    ax.set_ylabel("Density")
    ax.set_title(f"Ratio distribution (σ = {std:.2f} dex)")
    ax.legend()

    plt.suptitle("Radiation Field Comparison: Geometric estimate vs SKIRT", fontsize=14)
    plt.tight_layout()
    plt.savefig("compare_rf.png", dpi=150, bbox_inches='tight')
    print("\nSaved compare_rf.png")

    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Median ratio (geometric/SKIRT): {med_ratio:.2f}")
    print(f"  Median log ratio: {np.median(log_ratio_all):.2f} dex")
    print(f"  Scatter (σ): {std:.2f} dex")
    print(f"  16th–84th percentile ratio: "
          f"{10**np.percentile(log_ratio_all, 16):.2f} – "
          f"{10**np.percentile(log_ratio_all, 84):.2f}")


if __name__ == "__main__":
    main()

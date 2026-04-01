"""
Compute per-gas-cell 7.7 um PAH emissivity from an Arepo/SMUGGLE snapshot.

Physics:
  j_7.7(i) = C_7.7 * M_dust(i) * U(i)
where M_dust = 0.3 * Z * M_gas (matching SKIRT massFraction),
U is the local radiation field from nearby young stars,
and C_7.7 encodes the Draine & Li (2007) PAH emission per dust mass.

Output: HDF5 file with per-cell emissivity, dust mass, radiation field, etc.
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
import time as timer

# ── Paths ──────────────────────────────────────────────────────────────────
SNAP_PATH  = "../I5_output/snap_190.hdf5"
OUT_PATH   = "emissivity_snap190.h5"
BPASS_PATH = "/n/home10/asmith/colt/tables/bpass-spectra-bin-imf_chab100.hdf5"

# ── Physical constants (cgs) ───────────────────────────────────────────────
c_cgs    = 2.998e10       # speed of light, cm/s
pc_cm    = 3.0857e18      # parsec in cm
kpc_cm   = 3.0857e21      # kiloparsec in cm
Msun_g   = 1.989e33       # solar mass in grams
m_H      = 1.673e-24      # hydrogen mass in grams
u_ISRF   = 8.64e-13       # Mathis (1983) UV energy density, erg/cm^3

# Draine & Li (2007) dust model parameters
# Grain population mass fractions (M_dust / M_H) from DraineLiDustMix.cpp:
#   Silicate: 7.64e-3, Graphite: 2.21e-3 + 1.66e-4, PAH: 2×2.485e-4
DL07_TOTAL_DUST_PER_MH = 1.0264e-2
DL07_PAH_PER_MH        = 4.97e-4
Q_PAH = DL07_PAH_PER_MH / DL07_TOTAL_DUST_PER_MH   # 0.0484

# Total IR power per H atom at U=1: ~1.1e-23 erg/s (DL07 Eq. 29)
# → per gram of dust: 1.1e-23 / (1.0264e-2 * m_H) = 641 erg/s/g
# 7.7 um PAH band fraction of total IR: ~4.5% (DL07 Table 3 at q_PAH~4.8%)
P_IR_PER_GRAM_U1 = 641.0      # erg/s/g at U=1
F_77 = 0.045                   # fraction of total IR in 7.7 um band
C_77_DL07 = P_IR_PER_GRAM_U1 * F_77 * Msun_g  # erg/s per Msun_dust at U=1 ≈ 5.74e34
# Calibration factor from validate.py: matches radial profile to SKIRT F770W face-on output.
# With three-tier hybrid U-field (FFT grid + KDTree near-field + adaptive softening):
#   zoom (10 kpc, face-on): α = 1.74
#   ic5332 (10 kpc, inc=27°): α = 1.63
#   fo (40 kpc, face-on): α = 4.48  (outer r>5 kpc drives higher value)
# Using median of zoom/ic5332 for the inner-galaxy science case.
# (Was 12.10 with old K=200 approach.)
ALPHA_CAL = 1.68
C_77 = C_77_DL07 * ALPHA_CAL   # calibrated: ≈ 9.64e34

# SKIRT ski file parameter: massFraction for VoronoiMeshMedium
DUST_MASS_FRACTION = 0.3

# Sobolev dust attenuation
KAPPA_FUV = 1000.0  # FUV dust opacity, cm^2/g (Draine 2003, averaged over 912-3000 A)

# BPASS FUV luminosity table (binary population, Chabrier IMF)
# Loaded at import time; spectra in erg/s/A per Msun, ages in Gyr, wavelengths in Angstrom
FUV_WL_MIN = 912.0    # Lyman limit
FUV_WL_MAX = 3000.0   # near-UV cutoff


def _load_bpass_fuv_grid(path):
    """Pre-integrate BPASS spectra over FUV (912-3000 A) → L_FUV(Z, age) grid."""
    with h5py.File(path, "r") as f:
        age_gyr = f["age"][:]            # (N_age,) in Gyr
        Z_grid  = f["metallicity"][:]    # (N_Z,)
        wl      = f["wavelength"][:]     # (N_wl,) in Angstrom
        spec    = f["spectra"][:]        # (N_Z, N_age, N_wl) erg/s/A/Msun

    fuv = (wl >= FUV_WL_MIN) & (wl <= FUV_WL_MAX)
    # Integrate over FUV band (1 A bins)
    L_fuv_grid = np.sum(spec[:, :, fuv], axis=2)  # (N_Z, N_age) erg/s/Msun
    age_myr = age_gyr * 1e3
    return Z_grid, age_myr, L_fuv_grid


_BPASS_Z, _BPASS_AGE_MYR, _BPASS_LFUV = _load_bpass_fuv_grid(BPASS_PATH)
# Build log-space interpolator for smooth behaviour across decades
_LOG_BPASS_Z   = np.log10(np.maximum(_BPASS_Z, 1e-10))
_LOG_BPASS_AGE = np.log10(_BPASS_AGE_MYR)
_LOG_BPASS_LFUV = np.log10(np.maximum(_BPASS_LFUV, 1e-50))
_BPASS_INTERP = RegularGridInterpolator(
    (_LOG_BPASS_Z, _LOG_BPASS_AGE), _LOG_BPASS_LFUV,
    method="linear", bounds_error=False, fill_value=None,
)

# Radiation field computation
K_NEAR      = 32       # nearest young stars for near-field correction
H_SOFT_KPC  = 0.01     # minimum softening length in kpc (10 pc)
AGE_CUT_MYR = 300.0    # only stars younger than this contribute significant UV
CHUNK_SIZE  = 50_000   # gas cells per batch

# FFT grid-based far-field
GRID_SIZE    = 512      # 3D grid cells per side
GRID_BOX_KPC = 60.0    # box size in kpc (matching SKIRT FOV)

# Adaptive softening
K_SOFT_NN = 9           # neighbors for adaptive softening (use 8th neighbor distance)


def load_snapshot(path):
    """Load gas and star data from Arepo snapshot."""
    print(f"Loading snapshot: {path}")
    with h5py.File(path, "r") as f:
        header_time = f["Header"].attrs["Time"]

        # Gas (PartType0)
        gas_pos  = f["PartType0/Coordinates"][:]       # kpc (code units)
        gas_mass = f["PartType0/Masses"][:] * 1e10      # Msun
        gas_Z    = f["PartType0/GFM_Metallicity"][:]    # absolute
        gas_temp = f["PartType0/InternalEnergy"][:]     # code units (need conversion)
        gas_rho  = f["PartType0/Density"][:] * 1e10 * Msun_g / kpc_cm**3  # g/cm^3
        gas_mu   = 4.0 / (1.0 + 3 * 0.76 + 4 * 0.76 * f["PartType0/ElectronAbundance"][:])
        # T = (gamma-1) * u * mu * m_H / k_B, code velocity^2 → (km/s)^2
        k_B_cgs = 1.381e-16
        gas_temp = (2.0 / 3.0) * gas_temp * 1e10 * gas_mu * m_H / k_B_cgs  # K

        # Stars (PartType4)
        star_pos  = f["PartType4/Coordinates"][:]
        star_mass = f["PartType4/Masses"][:] * 1e10
        star_Z    = f["PartType4/GFM_Metallicity"][:]     # absolute
        star_gage = f["PartType4/GFM_StellarFormationTime"][:]
    # Center on stellar center of mass
    com = np.average(star_pos, weights=star_mass, axis=0)
    gas_pos  -= com
    star_pos -= com

    # Stellar ages: code time units → Myr
    # 1 code time unit = 1 kpc / (1 km/s) = 3.0857e21 / 1e5 s = 3.0857e16 s = 977.8 Myr
    code_time_to_myr = kpc_cm / 1e5 / (3.156e7 * 1e6)  # 977.8
    star_ages_myr = (header_time - star_gage) * code_time_to_myr

    return (gas_pos, gas_mass, gas_Z, gas_temp, gas_rho,
            star_pos, star_mass, star_Z, star_ages_myr)


def stellar_fuv_luminosity(star_mass, star_ages_myr, star_Z):
    """Compute FUV luminosity (912-3000 Å) from stellar mass, age, and metallicity.

    Uses BPASS v2 binary population spectra (Chabrier IMF) integrated over 912-3000 A.
    Bilinear interpolation in log(Z)-log(age) space.
    """
    L_FUV = np.zeros_like(star_mass)

    valid = star_ages_myr > 0
    log_age = np.log10(np.clip(star_ages_myr[valid],
                               _BPASS_AGE_MYR[0], _BPASS_AGE_MYR[-1]))
    log_Z = np.log10(np.clip(star_Z[valid], _BPASS_Z[0], _BPASS_Z[-1]))

    pts = np.column_stack([log_Z, log_age])
    log_luv_per_msun = _BPASS_INTERP(pts)
    L_FUV[valid] = star_mass[valid] * 10**log_luv_per_msun

    return L_FUV


def compute_U_grid(star_pos, L_UV, gas_pos, grid_size=GRID_SIZE, box_kpc=GRID_BOX_KPC):
    """Compute radiation field U on a 3D grid via FFT convolution with 1/r² kernel.

    Deposits all stellar FUV luminosity onto a grid using CIC assignment,
    convolves with the optically-thin radiation kernel, and interpolates
    back to gas cell positions.
    """
    dx = box_kpc / grid_size  # kpc per cell
    dx_cm = dx * kpc_cm
    half_box = box_kpc / 2.0

    print(f"Computing grid-based U-field ({grid_size}³, {box_kpc} kpc, dx={dx*1e3:.1f} pc)...")
    t0 = timer.time()

    # ── CIC deposit stellar luminosity onto grid ──
    L_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    # Map star positions to grid coordinates (0-indexed, centered)
    sx = (star_pos[:, 0] + half_box) / dx - 0.5
    sy = (star_pos[:, 1] + half_box) / dx - 0.5
    sz = (star_pos[:, 2] + half_box) / dx - 0.5

    # CIC: distribute luminosity to 8 surrounding cells
    ix0 = np.floor(sx).astype(int)
    iy0 = np.floor(sy).astype(int)
    iz0 = np.floor(sz).astype(int)
    fx = sx - ix0
    fy = sy - iy0
    fz = sz - iz0

    for dix in (0, 1):
        wx = np.where(dix == 0, 1.0 - fx, fx)
        ix = ix0 + dix
        for diy in (0, 1):
            wy = np.where(diy == 0, 1.0 - fy, fy)
            iy = iy0 + diy
            for diz in (0, 1):
                wz = np.where(diz == 0, 1.0 - fz, fz)
                iz = iz0 + diz
                w = wx * wy * wz * L_UV
                # Clip to grid bounds
                valid = ((ix >= 0) & (ix < grid_size) &
                         (iy >= 0) & (iy < grid_size) &
                         (iz >= 0) & (iz < grid_size))
                np.add.at(L_grid, (ix[valid], iy[valid], iz[valid]), w[valid])

    print(f"  CIC deposit: {timer.time()-t0:.1f}s, "
          f"L_deposited/L_total = {L_grid.sum()/L_UV.sum():.4f}")

    # ── Build 1/r² kernel ──
    # The radiation energy density from a luminosity cell is u = L / (4π r² c)
    # We convolve L_grid with K(r) = 1 / (4π r² c) to get u(x).
    t1 = timer.time()
    half = grid_size // 2
    kx = np.fft.fftfreq(grid_size, d=1.0) * grid_size  # cell offsets
    ky = kx.copy()
    kz = kx.copy()
    rx, ry, rz = np.meshgrid(kx, ky, kz, indexing='ij')
    r2 = (rx**2 + ry**2 + rz**2) * dx_cm**2  # cm²

    # Regularize r=0: average of 1/r² over a cell volume
    # For a cube of side dx: <1/r²> ≈ 3/(2π dx²) × (4π) from solid angle → use 6/dx²
    kernel = np.where(r2 > 0, 1.0 / (4.0 * np.pi * r2 * c_cgs),
                      1.0 / (4.0 * np.pi * (dx_cm**2 / 6.0) * c_cgs))

    print(f"  Kernel built: {timer.time()-t1:.1f}s")

    # ── FFT convolve ──
    t2 = timer.time()
    L_fft = np.fft.rfftn(L_grid)
    K_fft = np.fft.rfftn(kernel)
    u_grid = np.fft.irfftn(L_fft * K_fft, s=(grid_size, grid_size, grid_size))
    # u_grid is in erg/cm³ (energy density)
    U_grid = np.maximum(u_grid, 0) / u_ISRF  # convert to Mathis units

    print(f"  FFT convolve: {timer.time()-t2:.1f}s")

    # ── Interpolate to gas cell positions ──
    t3 = timer.time()
    gx = (gas_pos[:, 0] + half_box) / dx - 0.5
    gy = (gas_pos[:, 1] + half_box) / dx - 0.5
    gz = (gas_pos[:, 2] + half_box) / dx - 0.5

    coords = np.array([gx, gy, gz])
    U_at_cells = map_coordinates(U_grid, coords, order=1, mode='constant', cval=0.0)

    print(f"  Interpolation: {timer.time()-t3:.1f}s")
    print(f"  Grid U-field: median={np.median(U_at_cells[U_at_cells>0]):.2f}, "
          f"max={np.max(U_at_cells):.1f}")
    print(f"  Total grid time: {timer.time()-t0:.1f}s")

    return U_at_cells


def compute_adaptive_softening(star_pos_young):
    """Compute per-star adaptive softening from local inter-star spacing.

    Each star's softening is half the distance to its 8th nearest star neighbor,
    clipped to [H_SOFT_KPC, 0.5] kpc.
    """
    tree = cKDTree(star_pos_young)
    d_nn, _ = tree.query(star_pos_young, k=K_SOFT_NN)  # k=9, first is self (d=0)
    h_star = 0.5 * d_nn[:, K_SOFT_NN - 1]  # half-distance to 8th neighbor
    h_star = np.clip(h_star, H_SOFT_KPC, 0.5)
    print(f"Adaptive softening: median={np.median(h_star)*1e3:.1f} pc, "
          f"min={np.min(h_star)*1e3:.1f} pc, max={np.max(h_star)*1e3:.1f} pc")
    return h_star


def compute_U_near(gas_pos, star_pos_young, L_UV_young, h_star):
    """Compute near-field radiation field U using K-nearest young stars with adaptive softening.

    Uses Plummer softening: 1/(r² + h²) instead of hard floor.
    """
    N_gas = gas_pos.shape[0]
    N_young = star_pos_young.shape[0]
    K = min(K_NEAR, N_young)

    print(f"Building KDTree for {N_young} young stars...")
    tree = cKDTree(star_pos_young)

    U_field = np.zeros(N_gas, dtype=np.float64)
    n_chunks = (N_gas + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Computing near-field U for {N_gas} gas cells in {n_chunks} chunks (K={K})...")
    t0 = timer.time()

    for ic in range(n_chunks):
        i0 = ic * CHUNK_SIZE
        i1 = min(i0 + CHUNK_SIZE, N_gas)

        dist_kpc, idx = tree.query(gas_pos[i0:i1], k=K)
        if K == 1:
            dist_kpc = dist_kpc[:, np.newaxis]
            idx = idx[:, np.newaxis]

        dist_cm = dist_kpc * kpc_cm
        # Adaptive Plummer softening per star
        h_cm = h_star[idx] * kpc_cm  # (chunk, K)
        dist_soft_cm = np.sqrt(dist_cm**2 + h_cm**2)

        L_neighbors = L_UV_young[idx]
        u_contrib = L_neighbors / (4.0 * np.pi * dist_soft_cm**2 * c_cgs)
        U_field[i0:i1] = np.sum(u_contrib, axis=1) / u_ISRF

        if (ic + 1) % 10 == 0 or ic == n_chunks - 1:
            elapsed = timer.time() - t0
            print(f"  chunk {ic+1}/{n_chunks} ({elapsed:.1f}s)")

    return U_field


def sobolev_attenuation(gas_mass, gas_rho, gas_Z):
    """Compute Sobolev FUV dust attenuation factor per cell.

    In dense cells, dust absorbs incoming FUV before it can heat PAHs deeper
    in the cell. The effective attenuation factor is (1 - e^{-tau}) / tau,
    where tau = kappa_FUV * rho_dust * L_cell.
    """
    rho_dust = DUST_MASS_FRACTION * gas_Z * gas_rho                # g/cm^3
    L_cell = (gas_mass * Msun_g / gas_rho) ** (1.0 / 3.0)         # cm
    tau = KAPPA_FUV * rho_dust * L_cell
    f_att = np.where(tau > 1e-6, (1.0 - np.exp(-tau)) / tau, 1.0)
    return f_att, tau


def main():
    t_start = timer.time()

    # Load data
    (gas_pos, gas_mass, gas_Z, gas_temp, gas_rho,
     star_pos, star_mass, star_Z, star_ages_myr) = load_snapshot(SNAP_PATH)

    print(f"Gas cells: {len(gas_mass)}, Stars: {len(star_mass)}")

    # Stellar FUV luminosities from parametric SPS model
    L_UV = stellar_fuv_luminosity(star_mass, star_ages_myr, star_Z)

    # Filter to young stars (dominant UV sources)
    young = star_ages_myr < AGE_CUT_MYR
    # Also exclude wind particles (negative formation time → negative age)
    young &= star_ages_myr > 0
    N_young = np.sum(young)
    print(f"Young stars (age < {AGE_CUT_MYR} Myr): {N_young}")
    print(f"  UV luminosity fraction: {L_UV[young].sum()/L_UV[L_UV>0].sum():.1%}")

    star_pos_young = star_pos[young]
    L_UV_young = L_UV[young]

    # Compute radiation field: three-tier hybrid approach
    # Tier 1: FFT grid-based far-field (all stars, smooth)
    U_grid = compute_U_grid(star_pos_young, L_UV_young, gas_pos)

    # Tier 3: Adaptive softening for near-field stars
    h_star = compute_adaptive_softening(star_pos_young)

    # Tier 2: KDTree near-field with adaptive softening (K=32)
    U_near = compute_U_near(gas_pos, star_pos_young, L_UV_young, h_star)

    # Combine: max preserves near-field detail while grid fills in far-field
    U_field = np.maximum(U_grid, U_near)
    print(f"Combined U-field: median={np.median(U_field[U_field>0]):.2f}, "
          f"max={np.max(U_field):.1f}")
    frac_grid = np.mean(U_grid >= U_near)
    print(f"  Grid-dominated fraction: {frac_grid:.1%}")

    # Dust mass per cell (matching SKIRT's massFraction=0.3)
    M_dust = DUST_MASS_FRACTION * gas_Z * gas_mass  # Msun

    # Sobolev dust attenuation: reduce FUV in dense cells
    f_att, tau_fuv = sobolev_attenuation(gas_mass, gas_rho, gas_Z)

    # 7.7 um emissivity per cell
    j_77 = C_77 * M_dust * U_field * f_att  # erg/s

    # Summary statistics
    pos_U = U_field[U_field > 0]
    print(f"\nU-field: median={np.median(pos_U):.2f}, "
          f"mean={np.mean(pos_U):.2f}, max={np.max(pos_U):.1f}")
    print(f"Sobolev tau: median={np.median(tau_fuv):.3f}, "
          f"mean={np.mean(tau_fuv):.3f}, max={np.max(tau_fuv):.1f}")
    print(f"Attenuation f_att: median={np.median(f_att):.3f}, "
          f"min={np.min(f_att):.4f}")
    print(f"Total 7.7um luminosity: {j_77.sum():.3e} erg/s = {j_77.sum()/3.828e33:.2e} Lsun")
    print(f"C_77 = {C_77:.3e} erg/s/Msun_dust")

    # Save
    print(f"\nSaving to {OUT_PATH}")
    with h5py.File(OUT_PATH, "w") as f:
        g = f.create_group("gas")
        g.create_dataset("pos", data=gas_pos, dtype=np.float32)
        g.create_dataset("j_77", data=j_77, dtype=np.float64)
        g.create_dataset("M_dust", data=M_dust, dtype=np.float32)
        g.create_dataset("U_field", data=U_field, dtype=np.float32)
        g.create_dataset("temperature", data=gas_temp, dtype=np.float32)
        g.create_dataset("metallicity", data=gas_Z, dtype=np.float32)
        g.create_dataset("f_att", data=f_att, dtype=np.float32)
        g.create_dataset("tau_fuv", data=tau_fuv, dtype=np.float32)

        m = f.create_group("meta")
        m.attrs["C_77"] = C_77
        m.attrs["snapshot"] = SNAP_PATH
        m.attrs["K_near"] = K_NEAR
        m.attrs["age_cut_myr"] = AGE_CUT_MYR
        m.attrs["dust_mass_fraction"] = DUST_MASS_FRACTION
        m.attrs["q_PAH"] = Q_PAH
        m.attrs["h_soft_kpc"] = H_SOFT_KPC
        m.attrs["kappa_FUV"] = KAPPA_FUV
        m.attrs["sps_model"] = "BPASS v2 binary, Chabrier IMF"
        m.attrs["bpass_table"] = BPASS_PATH
        m.attrs["grid_size"] = GRID_SIZE
        m.attrs["grid_box_kpc"] = GRID_BOX_KPC
        m.attrs["U_method"] = "three-tier: FFT grid + KDTree near-field + adaptive softening"

    elapsed = timer.time() - t_start
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

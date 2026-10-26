"""Straight-ray (limited-angle CT style) imaging operator A' for pulse-echo SoS
from inter-transmit delays, as used by DenOiS (simplified, linearised model).

Unknown   u  = (1/c - 1/c_bf) * 1e6       [ns/mm]   on an (nz, nx) grid (row-major)
Measured  b  = tau_{k+1} - tau_k          [ns]      at the (nz, nx) recon pixel centres,
                                                    k = 0..npair-1, flattened (k, z, x)
"""
import numpy as np
import scipy.sparse as sp

from .acq import Geometry


def _ray_matrix(geo: Geometry, k, img_nz, img_nx, img_dz, img_dx, n_samp):
    """Travel-time-perturbation matrix for transmit k:
    rows = measurement pixels (geo.nz*geo.nx), cols = image pixels (img_nz*img_nx).
    Straight ray from its skin-entry point (x0, 0) to pixel centre p, bilinear weights."""
    Z, X = np.meshgrid(geo.z, geo.x, indexing='ij')
    zp = Z.ravel(); xp = X.ravel()
    x0, _ = geo.ray_entry(k, zp, xp)
    L = np.sqrt((xp - x0) ** 2 + zp ** 2)                          # [m]
    s = (np.arange(n_samp) + 0.5) / n_samp                         # midpoints
    zs = s[None, :] * zp[:, None]
    xs = x0[:, None] + s[None, :] * (xp - x0)[:, None]
    ds = (L / n_samp)[:, None] * np.ones_like(s)[None, :] * 1e3     # [mm]
    # fractional image indices (pixel centres at (i+0.5)*d, x centred like geo.x)
    img_x0 = geo.xc - (img_nx - 1) / 2 * img_dx
    fi = zs / img_dz - 0.5
    fj = (xs - img_x0) / img_dx
    fi = np.clip(fi, 0, img_nz - 1); fj = np.clip(fj, 0, img_nx - 1)
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, max(img_nz - 2, 0))
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, max(img_nx - 2, 0))
    wi = fi - i0; wj = fj - j0
    rows = np.repeat(np.arange(zp.size), n_samp)
    R, C, V = [], [], []
    for di, wI in ((0, 1 - wi), (1, wi)):
        for dj, wJ in ((0, 1 - wj), (1, wj)):
            R.append(rows); C.append(((i0 + di) * img_nx + (j0 + dj)).ravel()); V.append((wI * wJ * ds).ravel())
    M = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                      shape=(zp.size, img_nz * img_nx)).tocsr()
    M.sum_duplicates()
    return M


def build_operator(geo: Geometry, upsample=1, n_samp=None):
    """A' (upsample=1, recon grid) or the higher-resolution A_H (upsample=2):
    image grid is `upsample` times finer, measurement grid is always geo's 64x64.
    Returns scipy CSR of shape (npair*nz*nx, (nz*up)*(nx*up))."""
    nz, nx = geo.nz * upsample, geo.nx * upsample
    dz, dx = geo.dz / upsample, geo.dx / upsample
    n_samp = n_samp or int(np.ceil(geo.nz * geo.dz / min(dz, dx) * 1.5))
    T = [_ray_matrix(geo, k, nz, nx, dz, dx, n_samp) for k in range(geo.n_tx)]
    pairs = [T[k + 1] - T[k] for k in range(geo.npair)]
    return sp.vstack(pairs).tocsr().astype(np.float32)


def aperture_mask(geo: Geometry):
    """Aperture (undersampling) masks: measurement valid where both transmits of the
    pair insonify the pixel through their aperture.  (npair, nz, nx) bool."""
    Z, X = np.meshgrid(geo.z, geo.x, indexing='ij')
    ok = [geo.ray_entry(k, Z, X)[1] for k in range(geo.n_tx)]
    m = np.stack([ok[k] & ok[k + 1] for k in range(geo.npair)])
    m &= (Z >= geo.zmin_valid)[None]
    return m


def directivity_maps(geo: Geometry):
    """Prior channels hinting at depth- and angle-dependent expected variation:
    (npair, 2, nz, nx): normalised depth and the pair's mean steering angle."""
    Z, X = np.meshgrid(geo.z, geo.x, indexing='ij')
    d = Z / Z.max()
    out = []
    for k in range(geo.npair):
        x0a, _ = geo.ray_entry(k, Z, X); x0b, _ = geo.ray_entry(k + 1, Z, X)
        # mean incidence angle of the two rays -> expected sensitivity direction
        th = 0.5 * (np.arctan2(X - x0a, Z) + np.arctan2(X - x0b, Z))
        out.append(np.stack([d, th / (np.pi / 4)]))
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------- unit helpers

def sos_to_u(c, c_bf=1540.0):
    return (1.0 / c - 1.0 / c_bf) * 1e6


def u_to_sos(u, c_bf=1540.0):
    return 1.0 / (1.0 / c_bf + u * 1e-6)


def resample_gt(C, xC, zC, geo: Geometry, upsample=1):
    """Ground-truth SoS map (k-Wave grid) -> (box-averaged) recon grid."""
    from scipy.interpolate import RegularGridInterpolator
    nz, nx = geo.nz * upsample, geo.nx * upsample
    dz, dx = geo.dz / upsample, geo.dx / upsample
    sub = 5
    zz = (np.arange(nz * sub) + 0.5) * dz / sub
    xx = geo.xc + (np.arange(nx * sub) - (nx * sub - 1) / 2) * dx / sub
    f = RegularGridInterpolator((zC, xC), 1.0 / C, bounds_error=False, fill_value=None)
    Zq, Xq = np.meshgrid(zz, xx, indexing='ij')
    s = f(np.stack([Zq.ravel(), Xq.ravel()], -1)).reshape(nz * sub, nx * sub)
    s = s.reshape(nz, sub, nx, sub).mean((1, 3))                    # average slowness
    return (1.0 / s).astype(np.float32)

"""Random speed-of-sound media for training/testing (DenOiS datasets S and C).

quasi_random_sos : "quasi-random shapes and SoS distributions in [1400,1650] m/s,
                    some homogeneous, 50% blurred with random Gaussian kernels" (S, C1)
structured_sos   : well-structured circular / rectangular inclusions (C2 test set)
"""
import numpy as np
from scipy import ndimage

C_MIN, C_MAX = 1400.0, 1650.0


def _smooth_noise(rng, shape, sigma):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (n.std() + 1e-9)


def _blob_mask(rng, Z, X, zc, xc, rz, rx, rot, kind):
    zr = (Z - zc) * np.cos(rot) + (X - xc) * np.sin(rot)
    xr = -(Z - zc) * np.sin(rot) + (X - xc) * np.cos(rot)
    if kind == 'ellipse':
        return (zr / rz) ** 2 + (xr / rx) ** 2 <= 1
    if kind == 'rect':
        return (np.abs(zr) <= rz) & (np.abs(xr) <= rx)
    # irregular blob: ellipse with a smooth random radius perturbation
    th = np.arctan2(zr / rz, xr / rx)
    r = np.sqrt((zr / rz) ** 2 + (xr / rx) ** 2)
    k = rng.integers(2, 6)
    pert = 1 + sum(rng.uniform(0, 0.25) * np.cos(j * th + rng.uniform(0, 2 * np.pi)) for j in range(1, k + 1))
    return r <= pert


def quasi_random_sos(rng, nz, nx, dz, dx, layered_frac=0.3):
    """SoS map (nz, nx) [m/s] on a grid with pixel size (dz, dx) [m]."""
    Z, X = np.meshgrid((np.arange(nz) + 0.5) * dz, (np.arange(nx) - (nx - 1) / 2) * dx, indexing='ij')
    H, W = nz * dz, nx * dx
    c = np.full((nz, nx), rng.uniform(1430, 1620))
    if rng.random() < 0.5:                               # smooth background variation
        c += rng.uniform(0, 25) * _smooth_noise(rng, (nz, nx), rng.uniform(6, 20) * nz / 64)
    if rng.random() < layered_frac:                      # tissue-like layers from the skin down
        ztop = 0.0
        for _ in range(rng.integers(1, 4)):
            th = rng.uniform(2e-3, 12e-3)
            amp = rng.uniform(0, 2e-3); per = rng.uniform(8e-3, 40e-3); ph = rng.uniform(0, 2 * np.pi)
            zb = ztop + th + amp * np.sin(2 * np.pi * X / per + ph)
            c = np.where((Z >= ztop) & (Z < zb), rng.uniform(1420, 1620), c)
            ztop = ztop + th
    n_inc = rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.25, 0.15, 0.10])
    for _ in range(n_inc):
        kind = rng.choice(['ellipse', 'blob', 'rect'], p=[0.5, 0.35, 0.15])
        rz = rng.uniform(2e-3, 10e-3); rx = rz * rng.uniform(0.5, 2.0)
        m = _blob_mask(rng, Z, X, rng.uniform(3e-3, H - 2e-3), rng.uniform(-W / 2, W / 2), rz, rx,
                       rng.uniform(0, np.pi), kind)
        c = np.where(m, rng.uniform(C_MIN, C_MAX), c)
    if rng.random() < 0.5:
        c = ndimage.gaussian_filter(c, rng.uniform(0.5, 3.0) * nz / 64)
    return np.clip(c, C_MIN, C_MAX).astype(np.float32)


def structured_sos(rng, nz, nx, dz, dx):
    """Single circular or rectangular inclusion, positive or negative contrast (C2)."""
    Z, X = np.meshgrid((np.arange(nz) + 0.5) * dz, (np.arange(nx) - (nx - 1) / 2) * dx, indexing='ij')
    H, W = nz * dz, nx * dx
    bg = rng.uniform(1470, 1580)
    dc = rng.choice([-1, 1]) * rng.uniform(20, 70)
    kind = rng.choice(['ellipse', 'rect'])
    r = rng.uniform(3e-3, 7e-3)
    m = _blob_mask(rng, Z, X, rng.uniform(0.35 * H, 0.75 * H), rng.uniform(-W / 4, W / 4), r,
                   r * (1 if kind == 'ellipse' else rng.uniform(0.7, 1.4)), 0.0, kind)
    return np.clip(np.where(m, bg + dc, bg), C_MIN, C_MAX).astype(np.float32)


def random_echo(rng, sos):
    """Echogenicity map: smooth log-normal texture + inclusion-specific contrast."""
    nz, nx = sos.shape
    e = np.exp(0.3 * _smooth_noise(rng, (nz, nx), 4 * nz / 64))
    if rng.random() < 0.5:
        lv = np.round((sos - sos.min()) / 15)
        for v in np.unique(lv):
            if rng.random() < 0.5:
                e[lv == v] *= rng.uniform(0.4, 2.0)
    return e.astype(np.float32)


def downsample_sos(c_hi, f=2):
    """Average slowness over f x f blocks (hi-res GT -> recon grid GT)."""
    nz, nx = c_hi.shape
    s = (1.0 / c_hi).reshape(nz // f, f, nx // f, f).mean((1, 3))
    return (1.0 / s).astype(np.float32)

"""Evaluation metrics (DenOiS paper + thesis)."""
import numpy as np


def roi_mask(geo, x_lim=(-12e-3, 12e-3), z_lim=(0, 38.4e-3)):
    Z, X = np.meshgrid(geo.z, geo.x, indexing='ij')
    Xr = X - geo.xc
    return (Xr >= x_lim[0]) & (Xr <= x_lim[1]) & (Z >= z_lim[0]) & (Z <= z_lim[1])


def rmse(a, b, m=None):
    d = (a - b) if m is None else (a - b)[m]
    return float(np.sqrt(np.mean(d ** 2)))


def mae(a, b, m=None):
    d = (a - b) if m is None else (a - b)[m]
    return float(np.mean(np.abs(d)))


def inclusion_masks(gt, geo, thr=8.0, margin_mm=5.0):
    """Inclusion = connected region differing from the background median by > thr m/s;
    'background' = a margin ring (default 0.5 cm, as in DenOiS) around it."""
    from scipy import ndimage
    bg = np.median(gt)
    inc = np.abs(gt - bg) > thr
    if inc.sum() == 0:
        return None, None
    px = max(1, int(round(margin_mm * 1e-3 / min(geo.dz, geo.dx))))
    dil = ndimage.binary_dilation(inc, iterations=px)
    ring = dil & ~inc
    return inc, ring


def contrast(img, inc, ring):
    return float(np.median(img[inc]) - np.median(img[ring]))


def ace(rec, gt, inc, ring):
    return abs(contrast(rec, inc, ring) - contrast(gt, inc, ring))


def gcnr(img, inc, ring, bins=100):
    lo, hi = np.percentile(np.concatenate([img[inc], img[ring]]), [0, 100])
    h1, e = np.histogram(img[inc], bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(img[ring], bins=bins, range=(lo, hi), density=True)
    return float(1 - np.sum(np.minimum(h1, h2)) * (e[1] - e[0]))


def all_metrics(rec, gt, geo, roi=None):
    roi = roi_mask(geo) if roi is None else roi
    out = dict(RMSE=rmse(rec, gt, roi), MAE=mae(rec, gt, roi))
    inc, ring = inclusion_masks(gt, geo)
    if inc is not None and inc.sum() > 4 and ring.sum() > 4:
        out.update(ACE=ace(rec, gt, inc, ring), dC=contrast(rec, inc, ring), gCNR=gcnr(rec, inc, ring))
    return out

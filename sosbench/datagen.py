"""Dataset generation for DenOiS.

S : analytic (ray-model) data — hi-res operator A_H on 128x128 ground truths,
    reconstruction / learning only sees A' on 64x64 (discretisation mismatch).
C : wave-simulated data — split-step Born simulation -> DAS -> delay tracking
    (realistic noise, missing data and imaging-model mismatch).
Real : IMPACT channel data -> same DAS + tracking pipeline.

Every sample stores (float16 where large):
    x      (64,64)    ground-truth SoS [m/s]          (nan if unknown)
    b      (8,64,64)  observed delays b' [ns]
    w      (8,64,64)  validity (aperture & correlation), 0/1
    rho    (8,64,64)  tracking correlation (S: synthetic)
    l1,l2  (64,64)    L1-TV / L2-TV reconstructions from b' [m/s]
    Ax     (8,64,64)  model-consistent target A' x [ns]  (nan if unknown)
"""
import os, time
import numpy as np

from .acq import Geometry, beamform_pw, track_delays, observations_from_fsa, load_impact
from .raymodel import build_operator, aperture_mask, sos_to_u, u_to_sos, resample_gt
from .solvers import LinearOp, tv_recon, tv_irls, U_SCALE
from .phantoms import quasi_random_sos, structured_sos, random_echo, downsample_sos
from .wavesim import SimConfig, simulate_pw_iq, make_medium

RHO_THR = 0.2
# Analytical baselines (tuned once on dataset S, never on test data):
#   L1-TV : diagonally-preconditioned Chambolle-Pock, relative weight 0.003, 3000 its
#   L2-TV : IRLS + CG, weight 3e-5 (normalised units), 8 x 150 its
TV_CFG = dict(l1=dict(lam=0.003, iters=3000), l2=dict(lam=3e-5, outer=8, inner=150))


def set_fast_tv(factor=10):
    """Cheaper solver settings for CPU smoke tests."""
    TV_CFG['l1']['iters'] = max(50, TV_CFG['l1']['iters'] // factor)
    TV_CFG['l2']['inner'] = max(10, TV_CFG['l2']['inner'] // factor)


class Ops:
    """Bundle of operators on a backend."""
    def __init__(self, B, geo: Geometry, with_hi=True):
        self.B, self.geo = B, geo
        t0 = time.time()
        self.A = build_operator(geo)                       # A'  (scipy)
        self.op = LinearOp(B, self.A, geo.nz, geo.nx)
        self.AH = build_operator(geo, upsample=2) if with_hi else None
        self.opH = LinearOp(B, self.AH, 2 * geo.nz, 2 * geo.nx) if with_hi else None
        self.ap_mask = aperture_mask(geo)
        print(f'operators built in {time.time() - t0:.1f}s; s1={self.op.s1:.2f}', flush=True)

    def tv(self, b_ns, w, which):
        """Batched TV reconstruction. b_ns, w: numpy (nb, 8, 64, 64). Returns SoS numpy (nb,64,64)."""
        B, op = self.B, self.op
        nb = b_ns.shape[0]
        bn = B.asarray(op.b_to_n(b_ns.reshape(nb, -1).astype(np.float32)))
        ww = B.asarray(w.reshape(nb, -1).astype(np.float32))
        if which == 'l1':
            u = tv_recon(op, bn * ww, ww, lam=TV_CFG['l1']['lam'], p=1, iters=TV_CFG['l1']['iters'])
        else:
            c = TV_CFG['l2']
            u = tv_irls(op, bn * ww, ww, lam=c['lam'], p=2, outer=c['outer'], inner=c['inner'])
        return u_to_sos(B.to_numpy(u) * U_SCALE, self.geo.c_bf).astype(np.float32)

    def Ax(self, sos):
        """A' x for SoS maps (nb,64,64) -> (nb,8,64,64) [ns]."""
        u = sos_to_u(sos.reshape(sos.shape[0], -1), self.geo.c_bf).T
        return (self.A @ u).T.reshape(sos.shape[0], self.geo.npair, self.geo.nz, self.geo.nx).astype(np.float32)


def _save(path, **arrs):
    big = {k: (v.astype(np.float16) if v.dtype == np.float32 and v.ndim == 4 else v) for k, v in arrs.items()}
    np.savez(path, **big)


# ---------------------------------------------------------------------------- S
def gen_S(ops: Ops, n, path, seed=0, chunk=256, layered_frac=0.3):
    geo = ops.geo
    rng = np.random.default_rng(seed)
    X, Bo, W, L1, L2, AX = [], [], [], [], [], []
    for c0 in range(0, n, chunk):
        m = min(chunk, n - c0)
        hi = np.stack([quasi_random_sos(rng, 2 * geo.nz, 2 * geo.nx, geo.dz / 2, geo.dx / 2, layered_frac)
                       for _ in range(m)])
        lo = np.stack([downsample_sos(h) for h in hi])
        uH = sos_to_u(hi.reshape(m, -1), geo.c_bf).T
        b = (ops.AH @ uH).T.reshape(m, geo.npair, geo.nz, geo.nx).astype(np.float32)
        # measurement degradations: additive noise, random low-SNR holes, outliers
        sig = rng.uniform(0.5, 4.0, (m, 1, 1, 1))
        b = b + sig * rng.standard_normal(b.shape).astype(np.float32)
        holes = np.stack([_random_holes(rng, geo) for _ in range(m)])
        w = (ops.ap_mask[None] & ~holes).astype(np.float32)
        out = rng.random(b.shape) < 0.01
        b = np.where(out, rng.uniform(-60, 60, b.shape), b).astype(np.float32)
        L1.append(ops.tv(b, w, 'l1')); L2.append(ops.tv(b, w, 'l2'))
        X.append(lo); Bo.append(b * w); W.append(w); AX.append(ops.Ax(lo))
        print(f'S {c0 + m}/{n}', flush=True)
    _save(path, x=np.concatenate(X), b=np.concatenate(Bo), w=np.concatenate(W),
          l1=np.concatenate(L1), l2=np.concatenate(L2), Ax=np.concatenate(AX))


def _random_holes(rng, geo, p=0.6):
    from scipy import ndimage
    h = np.zeros((geo.npair, geo.nz, geo.nx), bool)
    if rng.random() < p:
        n = ndimage.gaussian_filter(rng.standard_normal((geo.npair, geo.nz, geo.nx)), (0, 4, 4))
        h = n > np.quantile(n, 1 - rng.uniform(0.05, 0.35))
    return h


# ---------------------------------------------------------------------------- C
def simulate_case(B, ops: Ops, cfg: SimConfig, rng, sos_hi):
    """Wave-simulate one medium; return observations (b', rho, w) and B-mode."""
    geo = ops.geo
    ext = (0.0, geo.nz * geo.dz, geo.x[0] - geo.dx / 2, geo.x[-1] + geo.dx / 2)
    echo = random_echo(rng, sos_hi)
    sos_s, refl = make_medium(rng, cfg, sos_hi, ext, echo)
    iq, t_out = simulate_pw_iq(B, cfg, sos_s, refl)
    imgs = beamform_pw(B, iq, t_out, geo)
    d, r = track_delays(B, imgs, geo)
    d, r = B.to_numpy(d).astype(np.float32), B.to_numpy(r).astype(np.float32)
    w = (ops.ap_mask & (r >= RHO_THR)).astype(np.float32)
    comp = B.to_numpy(B.abs(B.sum(imgs, 0)))
    return d, r, w, comp


def gen_C(B, ops: Ops, n, path, seed=1, kind='quasi', layered_frac=0.3, log_every=10):
    geo = ops.geo
    cfg = SimConfig(geo, fc=geo.fc)
    rng = np.random.default_rng(seed)
    X, Bo, W, R = [], [], [], []
    t0 = time.time()
    for i in range(n):
        if kind == 'quasi':
            hi = quasi_random_sos(rng, 2 * geo.nz, 2 * geo.nx, geo.dz / 2, geo.dx / 2, layered_frac)
        else:
            hi = structured_sos(rng, 2 * geo.nz, 2 * geo.nx, geo.dz / 2, geo.dx / 2)
        d, r, w, _ = simulate_case(B, ops, cfg, rng, hi)
        X.append(downsample_sos(hi)); Bo.append(d * w); W.append(w); R.append(r)
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f'C[{kind}] {i + 1}/{n}  {el / (i + 1):.2f}s/case', flush=True)
    X = np.stack(X); Bo = np.stack(Bo); W = np.stack(W); R = np.stack(R)
    L1 = np.concatenate([ops.tv(Bo[i:i + 64], W[i:i + 64], 'l1') for i in range(0, n, 64)])
    L2 = np.concatenate([ops.tv(Bo[i:i + 64], W[i:i + 64], 'l2') for i in range(0, n, 64)])
    _save(path, x=X, b=Bo, w=W, rho=R, l1=L1, l2=L2, Ax=ops.Ax(X))


# ---------------------------------------------------------------------------- real
def real_case(B, ops: Ops, name, root='impact_data'):
    geo = ops.geo
    data = load_impact(name, root)
    if abs(data['fc'] - geo.fc) > 1:
        print(f'note: {name} fc={data["fc"] / 1e6} MHz differs from the training fc={geo.fc / 1e6} MHz')
    geo_r = Geometry(**{**_geo_kwargs(geo), 'elx': data['elx'], 'fc': data['fc']})
    d, r, bm = observations_from_fsa(data, geo_r, B)
    d = d.astype(np.float32); r = r.astype(np.float32)
    w = (ops.ap_mask & (r >= RHO_THR)).astype(np.float32)
    x = resample_gt(data['C'], data['xC'], data['zC'], geo_r) if data['has_gt'] else np.full((geo.nz, geo.nx), np.nan, np.float32)
    out = dict(name=name, x=x, b=d * w, w=w, rho=r, bmode=bm, geo=geo_r)
    out['l1'] = ops.tv(out['b'][None], w[None], 'l1')[0]
    out['l2'] = ops.tv(out['b'][None], w[None], 'l2')[0]
    return out


def _geo_kwargs(geo):
    keys = ['fc', 'c_bf', 'tx_mode', 'n_tx', 'angles_deg', 'dw_subap', 'dw_zv', 'nz', 'nx', 'dz', 'dx',
            'bf_sub_z', 'bf_sub_x', 'fnum_rx', 'zmin_valid', 'track_win_z', 'track_win_x']
    return {k: getattr(geo, k) for k in keys}

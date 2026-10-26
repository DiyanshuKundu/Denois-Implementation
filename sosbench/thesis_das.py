"""Thesis method 1 — differentiable straight-ray delay-and-sum with the common-midpoint
phase-error loss (DAS+CMPE; Simson et al. 2023/2024 dbua, ported to PyTorch as in Meng 2026).

    tau(e, p)   = |p - e| * mean_k s(x_k)          straight ray, slowness on a coarse control grid
    q_ij(p)     = chi_ij(p) * IQ_ij(tau_i + tau_j - t0) * exp(i 2pi fc (tau_i + tau_j))
    I_ab(p)     = sum_i A_ai sum_j B_bj q_ij(p)    17-element translated sub-apertures
    Gamma       = sum_{p in patch} I_ab conj(I_{a+1,b-1})       common mid-point pairs
    L_RPE       = mean_valid log(1 + beta^2 angle(Gamma)^2)     (coherence-gated, beta=100)
    J           = L_RPE + lam * (mean (d_x c)^2 + mean (d_z c)^2)
Adam on the control-grid SoS (lr 10 m/s), 150 iterations from 1540 m/s.
"""
import time
import numpy as np


def fsa_iq(scat, t, fc, bw, fs_out=32e6, t_max=60e-6, tx_chunk=16):
    """FSA RF -> baseband IQ (ntx, nrx, nt_out)."""
    nt, nrx, ntx = scat.shape
    fs = 1 / np.mean(np.diff(t))
    f = np.fft.rfftfreq(nt, 1 / fs)
    band = np.where((f > max(fc - bw / 2, 0.2e6)) & (f < fc + bw / 2))[0]
    fb = f[band]
    nt_out = int(np.ceil((t_max - t[0]) * fs_out))
    t_out = t[0] + np.arange(nt_out) / fs_out
    E = np.exp(2j * np.pi * fb[:, None] * (t_out[None, :] - t[0]) - 2j * np.pi * fc * t_out[None, :]).astype(np.complex64)
    iq = np.zeros((ntx, nrx, nt_out), np.complex64)
    for c0 in range(0, ntx, tx_chunk):
        S = np.fft.rfft(scat[:, :, c0:c0 + tx_chunk], axis=0)[band].astype(np.complex64)   # (nf, nrx, ntc)
        iq[c0:c0 + tx_chunk] = np.einsum('frt,fn->trn', S, E, optimize=True) * (2.0 / nt)
    return iq, t_out


class DASSetup:
    def __init__(self, data, fc=None, bw=None, ctrl_shape=(31, 19), ctrl_x=(-12e-3, 12e-3), ctrl_z=(0, 38.4e-3),
                 n_patch=17, patch_k=5, patch_dx=None, pe_x=(-12.7e-3, 12.7e-3), pe_z=(5e-3, 40e-3),
                 subap=17, sub_stride=8, fnum=0.75, fs_out=32e6, n_ray=32):
        fc = fc or data['fc']; bw = bw or data['bw']
        self.fc = fc
        elx = data['elx']; self.elx = elx.astype(np.float32); self.xc = float(elx.mean())
        self.iq, self.t_out = fsa_iq(data['scat'], data['t'], fc, bw, fs_out, t_max=2.3 * pe_z[1] / 1400 + 3e-6)
        self.fs = fs_out
        lam = 1540 / fc
        pdx = patch_dx or lam / 2
        cx = self.xc + np.linspace(pe_x[0], pe_x[1], n_patch); cz = np.linspace(pe_z[0], pe_z[1], n_patch)
        off = (np.arange(patch_k) - (patch_k - 1) / 2) * pdx
        PZ, PX = np.meshgrid(cz, cx, indexing='ij')                                   # (np, np)
        self.px = (PX.reshape(-1, 1, 1) + off[None, None, :]).repeat(patch_k, 1).reshape(-1).astype(np.float32)
        self.pz = (PZ.reshape(-1, 1, 1) + off[None, :, None]).repeat(patch_k, 2).reshape(-1).astype(np.float32)
        self.n_patches = n_patch * n_patch; self.patch_px = patch_k * patch_k
        nel = len(elx)
        centers = np.arange(subap // 2, nel - subap // 2, sub_stride)
        A = np.zeros((len(centers), nel), np.float32)
        for k, c in enumerate(centers): A[k, c - subap // 2: c + subap // 2 + 1] = 1
        self.A = A
        self.ctrl_shape = ctrl_shape
        self.ctrl_x = self.xc + np.linspace(ctrl_x[0], ctrl_x[1], ctrl_shape[1])
        self.ctrl_z = np.linspace(ctrl_z[0], ctrl_z[1], ctrl_shape[0])
        self.fnum = fnum; self.n_ray = n_ray


def _tof(B, S, c_ctrl, ex, px, pz):
    """Straight-ray travel time element->pixel. ex: (nel,), px/pz: (P,) -> (nel, P) [s]."""
    nzc, nxc = S.ctrl_shape
    s_ctrl = 1.0 / c_ctrl
    k = (B.arange(S.n_ray) + 0.5) / S.n_ray                                         # (K,)
    xs = ex[:, None, None] + k[None, None, :] * (px[None, :, None] - ex[:, None, None])
    zs = k[None, None, :] * pz[None, :, None] + 0 * ex[:, None, None]
    fx = (xs - float(S.ctrl_x[0])) / float(S.ctrl_x[1] - S.ctrl_x[0])
    fz = (zs - float(S.ctrl_z[0])) / float(S.ctrl_z[1] - S.ctrl_z[0])
    sv = B.interp2(s_ctrl, fz, fx)                                                  # (nel, P, K)
    L = B.sqrt((px[None, :] - ex[:, None]) ** 2 + pz[None, :] ** 2)
    return L * B.mean(sv, axis=2)


def _subap_images(B, S, iq, tau, ex, px, pz, tx_idx):
    """For transmit elements tx_idx: S_ib(p) = sum_j B_bj q_ij(p). Returns (len(tx_idx), nsub, P)."""
    nt = iq.shape[-1]; t0 = float(S.t_out[0])
    Bw = B.asarray(S.A)                                                              # rx sub-aperture weights
    chi_rx = (B.abs(px[None, :] - ex[:, None]) * 2 * S.fnum <= pz[None, :])         # (nrx, P)
    chi_rx = chi_rx.astype(np.float32) if B.name == 'numpy' else chi_rx.float()
    out = []
    for i in tx_idx:
        T = tau[i][None, :] + tau                                                      # (nrx, P)
        idx = (T - t0) * S.fs
        i0 = B.clip(B.floor(idx), 0, nt - 2); w = idx - i0
        i0l = i0.astype(np.int64) if B.name == 'numpy' else i0.long()
        sig = iq[i]                                                                    # (nrx, nt)
        s0 = B.take_along(sig, i0l, 1); s1 = B.take_along(sig, i0l + 1, 1)
        chi_tx = (B.abs(px - float(S.elx[i])) * 2 * S.fnum <= pz)
        chi_tx = chi_tx.astype(np.float32) if B.name == 'numpy' else chi_tx.float()
        q = (s0 * (1 - w) + s1 * w) * B.polar(2 * np.pi * S.fc * T) * chi_rx * chi_tx[None, :]
        out.append(B.einsum('bj,jp->bp', Bw.astype(np.complex64) if B.name == 'numpy' else Bw.to(q.dtype), q))
    return B.stack(out, 0)


def cmpe_loss(B, S, I, beta=100.0, eta=0.5):
    """I: (nsub, nsub, P) sub-aperture images.  Common-midpoint pairs (a,b)~(a+1,b-1)."""
    ns = I.shape[0]
    Ip = I.reshape(ns, ns, S.n_patches, S.patch_px)
    X1 = Ip[:-1, 1:]; X2 = Ip[1:, :-1]
    G = B.sum(X1 * B.conj(X2), axis=-1)
    E1 = B.sum(B.abs(X1) ** 2, axis=-1); E2 = B.sum(B.abs(X2) ** 2, axis=-1)
    rho2 = B.abs(G) ** 2 / (E1 * E2 + 1e-30)
    valid = rho2 > eta ** 2
    Gs = B.where(valid, G, 0 * G + 1)                    # avoid angle'(0) = NaN in masked entries
    dphi = B.angle(Gs)
    vf = valid.astype(np.float32) if B.name == 'numpy' else valid.float()
    L = B.sum(B.log1p((beta * dphi) ** 2) * vf) / (B.sum(vf) + 1e-9)
    pe = B.sum(B.abs(dphi) * vf) / (B.sum(vf) + 1e-9)
    return L, pe, B.sum(vf)


def das_cmpe_forward_numpy(S: DASSetup, c_ctrl):
    from .backend import get_backend
    B = get_backend('numpy')
    ex = S.elx; px, pz = S.px, S.pz
    tau = _tof(B, S, c_ctrl.astype(np.float32), ex, px, pz)
    Sib = _subap_images(B, S, S.iq, tau, ex, px, pz, range(len(ex)))                 # (ntx, nsub, P)
    I = np.einsum('ai,ibp->abp', S.A.astype(np.complex64), Sib)
    return cmpe_loss(B, S, I)


def run_das_cmpe(S: DASSetup, device='cuda:0', iters=150, lr=10.0, lam=1e-3, c_init=1540.0, tx_chunk=8,
                 beta=100.0, eta=0.5, log=print):
    import torch
    from torch.utils.checkpoint import checkpoint
    from .backend import get_backend
    B = get_backend('torch', device)
    dev = torch.device(device)
    iq = torch.from_numpy(S.iq).to(dev)
    ex = torch.from_numpy(S.elx).to(dev); px = torch.from_numpy(S.px).to(dev); pz = torch.from_numpy(S.pz).to(dev)
    A = torch.from_numpy(S.A).to(dev).to(torch.complex64)
    c = torch.full(S.ctrl_shape, c_init, device=dev, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    nel = len(S.elx)
    torch.cuda.reset_peak_memory_stats(dev)
    hist = []; t0 = time.time()

    def chunk(tau, i0, i1):
        return _subap_images(B, S, iq, tau, ex, px, pz, range(i0, i1))

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        tau = _tof(B, S, c, ex, px, pz)                                   # (nel, P), differentiable
        I = 0
        for i0 in range(0, nel, tx_chunk):
            i1 = min(nel, i0 + tx_chunk)
            Sib = checkpoint(chunk, tau, i0, i1, use_reentrant=False)
            I = I + torch.einsum('ai,ibp->abp', A[:, i0:i1], Sib)
        L, pe, nv = cmpe_loss(B, S, I, beta, eta)
        reg = ((c[:, 1:] - c[:, :-1]) ** 2).mean() + ((c[1:] - c[:-1]) ** 2).mean()
        J = L + lam * reg
        J.backward()
        opt.step()
        with torch.no_grad():
            c.clamp_(1350, 1700)
        hist.append(dict(it=it, loss=float(L), pe=float(pe), n_valid=float(nv)))
        if it % 10 == 0 or it == iters - 1:
            log(f'  DAS+CMPE {it + 1}/{iters}  L_RPE={float(L):.3f}  PE={float(pe):.4f} rad  valid={int(nv)}  {time.time() - t0:.0f}s')
    return dict(c_ctrl=c.detach().cpu().numpy(), ctrl_z=S.ctrl_z, ctrl_x=S.ctrl_x, history=hist,
                time_per_iter=(time.time() - t0) / iters, peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)


def ctrl_to_grid(c_ctrl, ctrl_z, ctrl_x, z, x):
    """Bilinear interpolation of the control-grid SoS onto an evaluation grid."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((ctrl_z, ctrl_x), c_ctrl, bounds_error=False, fill_value=None)
    Z, X = np.meshgrid(np.clip(z, ctrl_z[0], ctrl_z[-1]), np.clip(x, ctrl_x[0], ctrl_x[-1]), indexing='ij')
    return f(np.stack([Z.ravel(), X.ravel()], -1)).reshape(Z.shape).astype(np.float32)

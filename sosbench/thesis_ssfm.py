"""Thesis method 2 — Fourier split-step (SSFM) wavefield model with reflection-matrix
focusing-quality optimisation (Meng 2026, following Ali et al. IMPACT and Lambert et al.).

Forward (per transmit element i, frequency f), IMPACT conventions:
    rx_i(x,0,f) = apod * P_i(x,f) e^{-i2pi f t0}  (nearest-neighbour embedding)
    tx_i(x,0,f) = delta(x - x_i)
    rx <- dH  * W IFFT[ H  * FFT[W rx]]          H = exp(+i2pi kz dz), dH = exp(+i2pi f (s - s_mean) dz)
    tx <- dH* * W IFFT[ H* * FFT[W tx]]
Focused reflection matrix at depth z (confocal band |x_in - x_out| <= w):
    R(x_in, x_out) = sum_{i,f} tx_i(x_in) conj(rx_i(x_out))
    F_RM = < |R(x,x)|^2 / sum_{|d|<=w} |R(x+d,x)|^2 >_ROI          (maximised)
Objective: -F_RM + aniso-TV(slowness);  slowness = s_m(z) + ds(z,x), two stages
(50 depth-only + 100 joint iterations), Adam, partial 3x3 median filter (blend 0.5).

Memory: the propagation is checkpointed over depth chunks, frequencies are
strided and transmit elements can be split across two GPUs (T4-friendly)."""
import time
import numpy as np


def opt_apod(nel, nlags=1, tol=1e-12, seed=0):
    """Short-lag-coherence optimal apodisation (port of IMPACT optApod)."""
    L = np.diag(2.0 * nlags * np.ones(nel))
    for lag in range(1, nlags + 1):
        L -= np.diag(np.ones(nel - lag), k=lag) + np.diag(np.ones(nel - lag), k=-lag)
    a = np.random.default_rng(seed).standard_normal(nel)
    prev, cur = 0.0, 1.0
    for _ in range(200):
        prev = cur
        a = np.abs(a) / np.linalg.norm(a)
        M = np.block([[L, a[:, None]], [a[None, :], np.zeros((1, 1))]])
        sol = np.linalg.solve(M, np.concatenate([np.zeros(nel), [1.0]]))
        a = sol[:nel]; cur = sol[-1]
        if abs(np.linalg.norm(a) - 1) < tol and abs((cur - prev) / cur) < tol:
            break
    return a


class SSFMSetup:
    """Pre-computed boundary wavefields and grids (numpy)."""
    def __init__(self, data, fc=None, bw=None, dov=40e-3, upsamp_x=2, nx0=None, f_stride=4, tx_stride=1,
                 c0=1540.0, ord_aa=50, roi_x=(-12e-3, 12e-3), roi_z=(2e-3, 38.4e-3), confocal_w=None):
        fc = fc or data['fc']; bw = bw or data['bw']
        elx = data['elx']; t = data['t']; scat = data['scat']
        nt, nrx, ntx = scat.shape
        pitch = float(np.mean(np.diff(elx)))
        self.xc = float(elx.mean())
        nx0 = nx0 or int(np.ceil((elx.max() - elx.min()) / pitch)) + 33            # ~aperture + margin
        self.x = self.xc + (pitch / upsamp_x) * (np.arange(upsamp_x * nx0) - (upsamp_x * nx0 - 1) / 2)
        lam = c0 / fc
        self.dz = lam / 2
        self.z = np.arange(int(round(dov / self.dz))) * self.dz
        xmax = (np.max(np.abs(elx - self.xc)) + np.max(np.abs(self.x - self.xc))) / 2
        self.aawin = (1 / np.sqrt(1 + ((self.x - self.xc) / xmax) ** ord_aa)).astype(np.float32)
        fs = 1 / np.mean(np.diff(t))
        f = np.fft.rfftfreq(nt, 1 / fs)
        band = np.where((f >= fc - bw / 2) & (f <= fc + bw / 2) & (f > 0))[0][::f_stride]
        self.f = f[band].astype(np.float32)
        tx_sel = np.arange(0, ntx, tx_stride)
        apod = opt_apod(nrx, 1)
        # receive boundary: (nx, nf, ntx_sel)
        P = np.fft.rfft(scat[:, :, tx_sel], axis=0)[band]                          # (nf, nrx, ntxs)
        P = P * np.exp(-2j * np.pi * self.f[:, None, None] * t[0]) * apod[None, :, None]
        idx = np.clip(np.round((self.x - elx[0]) / pitch).astype(int), 0, nrx - 1)
        inside = (self.x >= elx.min() - pitch / 2) & (self.x <= elx.max() + pitch / 2)
        rx = np.transpose(P, (1, 0, 2))[idx] * inside[:, None, None]
        self.rx0 = rx.astype(np.complex64)
        tx = np.zeros((len(self.x), len(self.f), len(tx_sel)), np.complex64)
        for k, i in enumerate(tx_sel):
            tx[np.argmin(np.abs(self.x - elx[i])), :, k] = 1.0
        self.tx0 = tx
        dxg = self.x[1] - self.x[0]
        self.confocal_w = confocal_w or max(2, int(round(1.5 * lam / dxg)))       # ~1.5 wavelengths
        Xr = self.x - self.xc
        self.roi_cols = np.where((Xr >= roi_x[0]) & (Xr <= roi_x[1]))[0]
        self.roi_rows = np.where((self.z >= roi_z[0]) & (self.z <= roi_z[1]))[0]
        self.c0 = c0
        print(f'SSFM grid: nx={len(self.x)} nz={len(self.z)} nf={len(self.f)} ntx={len(tx_sel)} '
              f'confocal w={self.confocal_w}  field {self.rx0.nbytes / 1e6:.0f} MB', flush=True)


def _prop(B, F, f, kx, smean, ds, dz, W, conj):
    """IMPACT propagate() for one field. F: (nx, nf, ntx)."""
    kz2 = (f[None, :] * smean) ** 2 - kx[:, None] ** 2                          # (nx, nf)
    ok = kz2 > 0
    okf = ok.astype(np.float32) if B.name == 'numpy' else ok.float()
    kz = B.sqrt(B.clip(kz2, 1.0, None))            # clamp >0 keeps d sqrt finite (branch-point regularisation)
    sgn = -1.0 if conj else 1.0
    H = B.polar(sgn * 2 * np.pi * kz * dz) * okf
    dH = B.polar(sgn * 2 * np.pi * f[None, :] * ds[:, None] * dz)               # (nx, nf)
    G = B.fft(W[:, None, None] * F, axis=0)
    return dH[:, :, None] * W[:, None, None] * B.ifft(H[:, :, None] * G, axis=0)


def _depth_terms(B, tx, rx, w, cols):
    """Confocal reflection-matrix focusing terms at one depth.
    Returns (num, den) per ROI column: |R(x,x)|^2 and sum_{|d|<=w} |R(x+d,x)|^2."""
    nx = tx.shape[0]
    num = None; den = 0
    for d in range(-w, w + 1):
        lo, hi = max(0, -d), min(nx, nx - d)
        Rd = B.sum(B.sum(tx[lo + d:hi + d] * B.conj(rx[lo:hi]), axis=2), axis=1)  # R(x+d, x), x in [lo,hi)
        full = B.pad(B.real(Rd) ** 2 + B.imag(Rd) ** 2, [(lo, nx - hi)])
        den = den + full
        if d == 0: num = full
    return num[cols], den[cols]


def ssfm_forward_numpy(setup: SSFMSetup, sos_grid, B=None):
    """Forward evaluation (no gradients) of the focusing criterion and complex image
    for a given SoS map on (setup.z, setup.x).  For validation on CPU."""
    from .backend import get_backend
    B = B or get_backend('numpy')
    s = 1.0 / sos_grid.astype(np.float32)
    kx = np.fft.fftfreq(len(setup.x), setup.x[1] - setup.x[0]).astype(np.float32)
    tx, rx = setup.tx0.copy(), setup.rx0.copy()
    img = np.zeros((len(setup.z), len(setup.x)), np.complex64)
    F = []
    for iz in range(len(setup.z) - 1):
        layer = 0.5 * (s[iz] + s[iz + 1]); sm = layer.mean(); ds = layer - sm
        rx = _prop(B, rx, setup.f, kx, sm, ds, setup.dz, setup.aawin, conj=False)
        tx = _prop(B, tx, setup.f, kx, sm, ds, setup.dz, setup.aawin, conj=True)
        img[iz + 1] = np.sum(tx * np.conj(rx), axis=(1, 2))
        if iz + 1 in set(setup.roi_rows):
            num, den = _depth_terms(B, tx, rx, setup.confocal_w, setup.roi_cols)
            F.append(np.mean(num / (den + 1e-12 * den.max() + 1e-30)))
    return float(np.mean(F)), img


# ------------------------------------------------------------------ torch optimisation
def run_ssfm(setup: SSFMSetup, devices=('cuda:0',), iters_depth=50, iters_joint=100, lr_m=3e-3, lr_d=6e-3,
             alpha_x=5.0, alpha_z=2.0, lam_tv=1.0, median_blend=0.5, c_init=None, chunk=12,
             c_bounds=(1350.0, 1700.0), log=print):
    """Two-stage SSFM optimisation. Slowness parameterised in s/km.
    Returns dict(sos (nz,nx) on setup grid, history, time_per_iter, peak_mem_gb)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from .backend import get_backend
    devs = [torch.device(d) for d in devices]
    Bs = {d: get_backend('torch', str(d)) for d in devs}
    nz, nx = len(setup.z), len(setup.x)
    c_init = c_init or setup.c0
    # split transmit elements across devices
    ntx = setup.tx0.shape[2]
    parts = np.array_split(np.arange(ntx), len(devs))
    TX0 = [torch.from_numpy(setup.tx0[:, :, p]).to(d) for p, d in zip(parts, devs)]
    RX0 = [torch.from_numpy(setup.rx0[:, :, p]).to(d) for p, d in zip(parts, devs)]
    fT = [torch.from_numpy(setup.f).to(d) for d in devs]
    kxT = [torch.from_numpy(np.fft.fftfreq(nx, setup.x[1] - setup.x[0]).astype(np.float32)).to(d) for d in devs]
    WT = [torch.from_numpy(setup.aawin).to(d) for d in devs]
    roi_rows = set(int(r) for r in setup.roi_rows)
    cols = torch.from_numpy(setup.roi_cols).to(devs[0])
    d0 = devs[0]
    s_m = torch.full((nz,), 1e3 / c_init, device=d0, requires_grad=True)                # s/km
    s_d = torch.zeros((nz, nx), device=d0, requires_grad=True)
    smin, smax = 1e3 / c_bounds[1], 1e3 / c_bounds[0]

    def chunk_fn(z0, z1, slow_km, *fields):
        """Propagate fields through depths z0..z1-1 and return updated fields + focusing sum/count."""
        k = len(devs)
        txs, rxs = list(fields[:k]), list(fields[k:])
        acc = torch.zeros((), device=d0); cnt = 0
        for iz in range(z0, z1):
            layer = 0.5 * (slow_km[iz] + slow_km[min(iz + 1, nz - 1)]) * 1e-3              # s/m
            sm = layer.mean(); ds = layer - sm
            nums, dens = 0, 0
            for j, d in enumerate(devs):
                B = Bs[d]
                lj, smj, dsj = layer.to(d), sm.to(d), ds.to(d)
                rxs[j] = _prop(B, rxs[j], fT[j], kxT[j], smj, dsj, setup.dz, WT[j], conj=False)
                txs[j] = _prop(B, txs[j], fT[j], kxT[j], smj, dsj, setup.dz, WT[j], conj=True)
            if iz + 1 in roi_rows:
                # reflection matrix needs the sum over ALL transmits before |.|^2
                w = setup.confocal_w
                num = None; den = 0
                for dd in range(-w, w + 1):
                    lo, hi = max(0, -dd), min(nx, nx - dd)
                    R = 0
                    for j, d in enumerate(devs):
                        R = R + (txs[j][lo + dd:hi + dd] * rxs[j][lo:hi].conj()).sum((1, 2)).to(d0)
                    full = torch.nn.functional.pad(R.real ** 2 + R.imag ** 2, (lo, nx - hi))
                    den = den + full
                    if dd == 0: num = full
                num, den = num[cols], den[cols]
                acc = acc + (num / (den + 1e-12 * den.detach().max() + 1e-30)).mean(); cnt += 1
        return (*txs, *rxs, acc, torch.tensor(float(cnt), device=d0))

    def objective(slow_km):
        fields = [t.clone() for t in TX0] + [r.clone() for r in RX0]
        tot = 0; n = 0
        for z0 in range(0, nz - 1, chunk):
            z1 = min(nz - 1, z0 + chunk)
            out = checkpoint(chunk_fn, z0, z1, slow_km, *fields, use_reentrant=False)
            fields = list(out[:-2]); tot = tot + out[-2]; n += out[-1].item()
        F_rm = tot / max(n, 1)
        tv = alpha_z * (slow_km[1:] - slow_km[:-1]).abs().mean() + alpha_x * (slow_km[:, 1:] - slow_km[:, :-1]).abs().mean()
        return -F_rm + lam_tv * tv, F_rm.detach()

    hist = []; t_start = time.time()
    for d in devs: torch.cuda.reset_peak_memory_stats(d)
    for stage, iters in (('depth', iters_depth), ('joint', iters_joint)):
        params = [{'params': [s_m], 'lr': lr_m}] + ([{'params': [s_d], 'lr': lr_d}] if stage == 'joint' else [])
        opt = torch.optim.Adam(params)
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            slow = s_m[:, None] + (s_d if stage == 'joint' else s_d.detach() * 0)
            loss, Frm = objective(slow)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if stage == 'joint':
                    full = (s_m[:, None] + s_d).clamp(smin, smax)
                    med = Bs[d0].median3(full)
                    full = median_blend * full + (1 - median_blend) * med
                    s_m.copy_(full.mean(1)); s_d.copy_(full - s_m[:, None])
                else:
                    s_m.clamp_(smin, smax)
            hist.append(dict(stage=stage, it=it, loss=float(loss), F=float(Frm)))
            if it % 10 == 0 or it == iters - 1:
                log(f'  SSFM {stage} {it + 1}/{iters}  F_RM={float(Frm):.4f}  loss={float(loss):.4f}  {time.time() - t_start:.0f}s')
    sos = (1e3 / (s_m[:, None] + s_d)).detach().cpu().numpy()
    n_it = iters_depth + iters_joint
    mem = max(torch.cuda.max_memory_allocated(d) for d in devs) / 1e9
    return dict(sos=sos, z=setup.z, x=setup.x, history=hist, time_per_iter=(time.time() - t_start) / n_it, peak_mem_gb=mem)

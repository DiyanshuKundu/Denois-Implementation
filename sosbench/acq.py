"""Acquisition side: data loading (IMPACT .mat files), plane-wave synthesis from
full-synthetic-aperture (FSA) data, plane-wave delay-and-sum beamforming and
inter-transmit displacement (delay) tracking.  Produces the DenOiS observations b'.
"""
import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- data

IMPACT_URL = 'https://github.com/rehmanali1994/IMPACT/releases/download/v1.0.0/'
IMPACT_FILES = {
    'abd3':    dict(file='AbdominalMap3.mat', var='scat', fc=8e6, bw=12e6, gt=True),
    'abd4':    dict(file='AbdominalMap4.mat', var='scat', fc=8e6, bw=12e6, gt=True),
    'phantom': dict(file='PhantomL12-5-50mm.mat', var='fsr_dataset_fund', fc=6e6, bw=6e6, gt=False),
    'rat':     dict(file='RatAbdomenL12-3v.mat', var='full_synth_data', fc=7.5e6, bw=9e6, gt=False),
}


def download_impact(names=('abd3', 'abd4', 'phantom', 'rat'), root='impact_data'):
    import os, urllib.request
    os.makedirs(root, exist_ok=True)
    for n in names:
        fn = IMPACT_FILES[n]['file']; dst = os.path.join(root, fn)
        if not os.path.exists(dst):
            print('downloading', fn, '...', flush=True)
            urllib.request.urlretrieve(IMPACT_URL + fn, dst)
    return root


def _read_mat73(path, keys):
    """MATLAB v7.3 (HDF5) reader. Uses h5py (Kaggle); falls back to pyfive."""
    out = {}
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            for k in keys:
                if k in f: out[k] = np.asarray(f[k]).T
        return out
    except ImportError:
        import sys; sys.path.insert(0, '/home/claude/ext')
        from matload import load
        d = load(path)
        return {k: d[k] for k in keys if k in d}


def load_impact(name, root='impact_data', n_center_elements=128):
    """Returns dict with
        scat  : (nt, nrx, ntx) float32 FSA channel data
        t     : (nt,) time axis [s]
        elx   : (nel,) element x positions [m]
        fc,bw : centre frequency / bandwidth [Hz]
        C,xC,zC (if ground truth available)
    For probes with more than `n_center_elements`, the central sub-aperture is
    kept so that every dataset shares the same 128-element geometry as the
    training simulations."""
    import os
    meta = IMPACT_FILES[name]
    d = _read_mat73(os.path.join(root, meta['file']), [meta['var'], 'time', 'rxAptPos', 'C', 'x', 'z'])
    scat = d[meta['var']].astype(np.float32)          # (nt, nrx, ntx)
    elx = d['rxAptPos'][:, 0].astype(np.float64)
    nel = elx.size
    if n_center_elements and nel > n_center_elements:
        s = (nel - n_center_elements) // 2
        sl = slice(s, s + n_center_elements)
        scat = scat[:, sl, sl]; elx = elx[sl]
    out = dict(name=name, scat=np.ascontiguousarray(scat), t=d['time'].ravel().astype(np.float64),
               elx=elx, fc=meta['fc'], bw=meta['bw'], has_gt=meta['gt'])
    if meta['gt']:
        out.update(C=d['C'].astype(np.float32), xC=d['x'].ravel(), zC=d['z'].ravel())
    return out


# ----------------------------------------------------------------------------- geometry

@dataclass
class Geometry:
    """Acquisition + reconstruction geometry shared by every part of DenOiS."""
    elx: np.ndarray                       # element x [m]
    fc: float = 8e6
    c_bf: float = 1540.0                  # beamforming / reference speed of sound
    tx_mode: str = 'pw'                   # 'pw' plane waves | 'dw' diverging waves (virtual sources)
    n_tx: int = 9
    angles_deg: tuple = tuple(np.linspace(-16, 16, 9))   # used when tx_mode == 'pw'
    dw_subap: int = 64                    # elements per diverging-wave sub-aperture
    dw_zv: float = 4e-3                   # virtual source depth behind the array [m]
    # reconstruction grid (DenOiS: 64x64)
    nz: int = 64
    nx: int = 64
    dz: float = 0.6e-3
    dx: float = 0.4e-3                    # 64*0.4 mm = 25.6 mm = aperture width
    # beamforming grid oversampling inside one reconstruction pixel
    bf_sub_z: int = 12
    bf_sub_x: int = 4
    fnum_rx: float = 1.0
    zmin_valid: float = 2e-3              # near-field cut for measurements
    track_win_z: int = 2                  # tracking window in recon pixels (axial)
    track_win_x: int = 2                  # tracking window in recon pixels (lateral)

    def __post_init__(self):
        self.elx = np.asarray(self.elx, dtype=np.float64)
        self.xc = float(self.elx.mean())
        self.angles = np.deg2rad(np.asarray(self.angles_deg, dtype=np.float64))
        self.z = (np.arange(self.nz) + 0.5) * self.dz                       # pixel centres
        self.x = self.xc + (np.arange(self.nx) - (self.nx - 1) / 2) * self.dx
        self.zb = (np.arange(self.nz * self.bf_sub_z) + 0.5) * self.dz / self.bf_sub_z
        self.xb = self.xc + (np.arange(self.nx * self.bf_sub_x) - (self.nx * self.bf_sub_x - 1) / 2) * self.dx / self.bf_sub_x
        lo, hi = self.elx.min(), self.elx.max()
        nel = len(self.elx)
        if self.tx_mode == 'pw':
            self.n_tx = len(self.angles)
            self.tx_delay = np.zeros((self.n_tx, nel)); self.tx_apod = np.ones((self.n_tx, nel))
            for a, th in enumerate(self.angles):
                xref = lo if np.sin(th) >= 0 else hi
                self.tx_delay[a] = (self.elx - xref) * np.sin(th) / self.c_bf
        else:
            # virtual sources evenly spread so that every sub-aperture fits in the array
            half = (self.dw_subap - 1) / 2 * np.mean(np.diff(self.elx))
            self.vx = np.linspace(lo + half, hi - half, self.n_tx)
            self.tx_delay = np.zeros((self.n_tx, nel)); self.tx_apod = np.zeros((self.n_tx, nel))
            for k, xv in enumerate(self.vx):
                sel = np.abs(self.elx - xv) <= half + 1e-9
                r = np.sqrt((self.elx - xv) ** 2 + self.dw_zv ** 2)
                self.tx_apod[k, sel] = 1.0
                self.tx_delay[k] = np.where(sel, (r - self.dw_zv) / self.c_bf, 0.0)
            self.dw_half = half
        self.npair = self.n_tx - 1

    def t_tx(self, k, Z, X, c=None):
        """Nominal transmit arrival time at points (Z, X) for transmit k (backend-agnostic)."""
        c = c or self.c_bf
        if self.tx_mode == 'pw':
            th = float(self.angles[k]); xref = self.elx.min() if np.sin(th) >= 0 else self.elx.max()
            return ((X - xref) * float(np.sin(th)) + Z * float(np.cos(th))) / c
        xv, zv = float(self.vx[k]), self.dw_zv
        return (((X - xv) ** 2 + (Z + zv) ** 2) ** 0.5 - zv) / c

    def ray_entry(self, k, Z, X):
        """Skin-entry x of the transmit ray reaching (Z, X), and whether it lies
        inside the insonifying aperture (numpy)."""
        if self.tx_mode == 'pw':
            x0 = X - Z * np.tan(self.angles[k])
            return x0, (x0 >= self.elx.min()) & (x0 <= self.elx.max())
        xv, zv = self.vx[k], self.dw_zv
        x0 = xv + (X - xv) * zv / (Z + zv)
        return x0, np.abs(x0 - xv) <= self.dw_half

    @property
    def extent_mm(self):
        return [1e3 * (self.x[0] - self.dx / 2), 1e3 * (self.x[-1] + self.dx / 2),
                1e3 * (self.z[-1] + self.dz / 2), 1e3 * (self.z[0] - self.dz / 2)]


def default_geometry(elx=None, **kw):
    """Training/evaluation geometry: 128-element, 0.2 mm pitch linear array (as in the
    IMPACT k-Wave data), 9 steered plane waves in [-16, 16] deg (8 transmit pairs ->
    8 delay maps), 64x64 reconstruction grid.  ('dw' = diverging waves is available
    but decorrelates strongly between neighbouring virtual sources on this probe.)"""
    if elx is None:
        elx = -12.56e-3 + 0.2e-3 * np.arange(128)
    args = dict(tx_mode='pw', n_tx=9, angles_deg=tuple(np.linspace(-16, 16, 9)), fc=8e6)
    args.update(kw)
    return Geometry(elx=elx, **args)


# ----------------------------------------------------------------------------- FSA -> plane waves

def fsa_to_pw_iq(scat, t, geo, fc, bw, fs_out=16e6, t_max=70e-6, rx_chunk=16):
    """Synthesise the geometry's transmits (plane or diverging waves) from FSA data
    in the frequency domain, and return baseband IQ channel data.
        scat: (nt, nrx, ntx) real; returns iq (n_tx, nrx, nt_out) complex64, t_out (nt_out,)"""
    nt, nrx, ntx = scat.shape
    fs = 1.0 / np.mean(np.diff(t))
    f = np.fft.rfftfreq(nt, 1 / fs)
    band = np.where((f > max(fc - bw / 2, 0.2e6)) & (f < fc + bw / 2))[0]
    fb = f[band]
    tau = geo.tx_delay[:, :ntx]
    Ptx = (np.exp(-2j * np.pi * fb[:, None, None] * tau[None, :, :]) * geo.tx_apod[None, :, :ntx]).astype(np.complex64)
    nt_out = int(np.ceil((t_max - t[0]) * fs_out))
    t_out = t[0] + np.arange(nt_out) / fs_out
    # baseband DFT synthesis matrix: spectrum referenced to t[0]
    E = np.exp(2j * np.pi * fb[:, None] * (t_out[None, :] - t[0])
               - 2j * np.pi * fc * t_out[None, :]).astype(np.complex64)                    # (nf, nt_out)
    iq = np.zeros((geo.n_tx, nrx, nt_out), np.complex64)
    for r0 in range(0, nrx, rx_chunk):
        S = np.fft.rfft(scat[:, r0:r0 + rx_chunk, :], axis=0)[band].astype(np.complex64)     # (nf, nrc, ntx)
        Y = np.einsum('frt,fat->afr', S, Ptx, optimize=True)                                 # (nang, nf, nrc)
        iq[:, r0:r0 + rx_chunk, :] = np.einsum('afr,fn->arn', Y, E, optimize=True) * (2.0 / nt)
    return iq, t_out


# ----------------------------------------------------------------------------- beamforming

def beamform_pw(B, iq, t_out, geo: Geometry, c=None, rx_chunk=32, ang_idx=None):
    """Plane/diverging-wave DAS with constant speed `c` on the fine grid (geo.zb, geo.xb).
    iq: (nang, nrx, nt) complex (backend array); returns (nang, nzb, nxb) complex."""
    c = c or geo.c_bf
    zb = B.asarray(geo.zb, np.float32)[:, None]; xb = B.asarray(geo.xb, np.float32)[None, :]
    elx = geo.elx
    t0 = float(t_out[0]); fs = 1.0 / float(t_out[1] - t_out[0]); nt = iq.shape[-1]
    ang_idx = range(geo.n_tx) if ang_idx is None else ang_idx
    out = []
    for a in ang_idx:
        t_tx = geo.t_tx(a, zb, xb, c)                                                    # (nz, nx)
        img = B.zeros((len(geo.zb), len(geo.xb)), B.cfloat)
        for r0 in range(0, len(elx), rx_chunk):
            ex = B.asarray(elx[r0:r0 + rx_chunk], np.float32)[:, None, None]
            dxr = xb[None] - ex
            t_rx = B.sqrt(dxr ** 2 + zb[None] ** 2) / c                                  # (nr, nz, nx)
            tau = t_tx[None] + t_rx
            apod = (B.abs(dxr) * 2 * geo.fnum_rx <= zb[None]).astype(np.float32) if B.name == 'numpy' \
                else (B.abs(dxr) * 2 * geo.fnum_rx <= zb[None]).float()
            idx = (tau - t0) * fs
            i0 = B.clip(B.floor(idx), 0, nt - 2)
            w = idx - i0
            i0l = i0.astype(np.int64) if B.name == 'numpy' else i0.long()
            sig = iq[a, r0:r0 + rx_chunk]                                                 # (nr, nt)
            nr = sig.shape[0]
            flat = i0l.reshape(nr, -1)
            s0 = B.take_along(sig, flat, 1).reshape(i0l.shape)
            s1 = B.take_along(sig, flat + 1, 1).reshape(i0l.shape)
            val = (s0 * (1 - w) + s1 * w) * B.polar(2 * np.pi * geo.fc * tau) * apod
            valid = (idx >= 0) & (idx <= nt - 1)
            val = val * (valid.astype(np.float32) if B.name == 'numpy' else valid.float())
            img = img + B.sum(val, axis=0)
        out.append(img)
    return B.stack(out, 0)


# ----------------------------------------------------------------------------- tracking

def _box_sum(B, x, kz, kx):
    """Centred box sum over (kz, kx) along the last two axes (zero padded).
    Padding is done by concatenation so it works for complex tensors on any backend."""
    def along(x, k, axis):
        n = x.shape[axis]
        before, after = k // 2 + 1, k - k // 2 - 1
        def sl(a, b):
            idx = [slice(None)] * x.ndim; idx[axis] = slice(a, b); return tuple(idx)
        parts = [0 * x[sl(0, before)], x]
        if after > 0: parts.append(0 * x[sl(0, after)])
        cs = B.cumsum(B.cat(parts, axis), axis)
        return cs[sl(k, k + n)] - cs[sl(0, n)]
    return along(along(x, kz, -2), kx, -1)


def track_delays(B, imgs, geo: Geometry, cycle_fix=False, max_lag=5):
    """Inter-transmit delays between consecutive frames (NCC-style tracking).
    1) fine estimate: phase of the zero-lag normalised complex cross-correlation in
       a sliding window  ->  precise but ambiguous modulo 1/fc (default output);
    2) [cycle_fix=True, for large delays e.g. diverging waves] coarse estimate: axial lag (in beamforming samples, +-max_lag, parabolic
       sub-sample refinement) maximising the windowed |cross-correlation|, converted
       to time with dt/dz = (cos(tx angle) + cos(rx angle))/c;
    3) the fine estimate is moved to the 1/fc cycle closest to the coarse one.
    imgs: (ntx, nzb, nxb) complex.  Returns
        delay_ns (npair, nz, nx)  : tau_{k+1} - tau_k  [ns]
        rho      (npair, nz, nx)  : normalised (zero-lag) correlation magnitude"""
    kz = geo.track_win_z * geo.bf_sub_z; kx = geo.track_win_x * geo.bf_sub_x
    a = imgs[:-1]; b = imgs[1:]
    nzb = a.shape[1]
    iz = B.asarray(np.arange(geo.nz) * geo.bf_sub_z + geo.bf_sub_z // 2)
    ix = B.asarray(np.arange(geo.nx) * geo.bf_sub_x + geo.bf_sub_x // 2)
    def sample(x): return x[:, iz][:, :, ix]
    cross0 = sample(_box_sum(B, b * B.conj(a), kz, kx))
    ea = sample(_box_sum(B, B.abs(a) ** 2, kz, kx)); eb = sample(_box_sum(B, B.abs(b) ** 2, kz, kx))
    rho = B.abs(cross0) / B.sqrt(ea * eb + 1e-30)
    fine = -B.angle(cross0) / (2 * np.pi * geo.fc)                              # [s]
    if not cycle_fix:                       # plane waves: |delay| << 1/(2 fc), no wrapping
        return fine * 1e9, rho
    # coarse lag search on |xcorr|
    mags = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            bs = B.cat([b[:, lag:], 0 * b[:, :lag]], axis=1) if lag > 0 else b
        else:
            bs = B.cat([0 * b[:, :(-lag)], b[:, :lag]], axis=1)
        mags.append(B.abs(sample(_box_sum(B, bs * B.conj(a), kz, kx))))
    M = B.stack(mags, 0)                                                        # (nlag, np, nz, nx)
    if B.name == 'numpy':
        k = M.argmax(0)
    else:
        k = M.argmax(0)
    kc = B.clip(k, 1, 2 * max_lag - 1)
    kcl = kc if B.name == 'numpy' else kc.long()
    take = lambda off: B.take_along(M, (kcl + off)[None], 0)[0]
    ym, y0, yp = take(-1), take(0), take(1)
    den = ym - 2 * y0 + yp
    frac = B.where(B.abs(den) > 1e-12, 0.5 * (ym - yp) / (den + 1e-30), 0 * den)
    frac = B.clip(frac, -0.5, 0.5)
    lag = (kc - max_lag) + frac                                                 # bf samples
    dzb = geo.dz / geo.bf_sub_z
    # dt/dz at the measurement points (mean of the two transmits' ray angles, rx ~ normal)
    Z, X = np.meshgrid(geo.z, geo.x, indexing='ij')
    cosk = []
    for k_ in range(geo.npair):
        x0a, _ = geo.ray_entry(k_, Z, X); x0b, _ = geo.ray_entry(k_ + 1, Z, X)
        ca = Z / np.sqrt((X - x0a) ** 2 + Z ** 2); cb = Z / np.sqrt((X - x0b) ** 2 + Z ** 2)
        cosk.append(0.5 * (ca + cb))
    dtdz = B.asarray(((np.stack(cosk) + 0.95) / geo.c_bf).astype(np.float32))
    coarse = lag * dzb * dtdz                                                   # [s]
    T = 1.0 / geo.fc
    n = B.floor((coarse - fine) / T + 0.5)
    delay_ns = (fine + n * T) * 1e9
    return delay_ns, rho


def observations_from_fsa(data, geo: Geometry, B=None, fs_out=16e6):
    """Full real-data pipeline: FSA -> 9 plane waves -> DAS -> tracking.
    Returns delay_ns (npair,nz,nx), rho (npair,nz,nx), bmode_db (nzb,nxb) (compounded)."""
    from .backend import get_backend
    B = B or get_backend('numpy')
    t_max = 2.2 * (geo.z[-1] + geo.dz) / 1400 + 3e-6
    iq, t_out = fsa_to_pw_iq(data['scat'], data['t'], geo, data['fc'], data['bw'], fs_out=fs_out, t_max=t_max)
    iq = B.asarray(iq)
    imgs = beamform_pw(B, iq, t_out, geo)
    d, r = track_delays(B, imgs, geo)
    comp = B.to_numpy(B.abs(B.sum(imgs, 0)))
    bmode = 20 * np.log10(comp / comp.max() + 1e-6)
    return B.to_numpy(d), B.to_numpy(r), bmode

"""Wave-based pulse-echo simulator (split-step Fourier / angular-spectrum Born model).

Plays the role of DenOiS' "complex in-silico" k-Wave dataset (C): it includes
diffraction, refraction-like phase-screen effects and finite-aperture physics that
the straight-ray operator A' ignores, but runs in ~1 s per case on a T4.

    downward:  T(z+dz) = S * IFFT_x[ H * FFT_x[ W T(z) ] ]           (transmit field)
    upward  :  U(z)    = S * IFFT_x[ H * FFT_x[ W U(z+dz) ] ] + T(z) r(z)   (Born echoes)
    H = exp(-i 2pi dz sqrt((f*smean)^2 - kx^2)),  S = exp(-i 2pi f dz (s - smean))
Delays correspond to exp(-i 2pi f tau) (numpy FFT convention), consistent with acq.py.
"""
import numpy as np

from .acq import Geometry


class SimConfig:
    def __init__(self, geo: Geometry, fc=8e6, frac_bw=0.75, dx_sim=None, dz_sim=None, x_span=38.4e-3,
                 z_max=None, fs_out=16e6, f_chunk=None, ord_aa=50, el_width_frac=0.9):
        self.geo = geo
        self.fc = fc
        pitch = float(np.mean(np.diff(geo.elx)))
        self.dx = dx_sim or pitch / 2
        self.dz = dz_sim or 0.1e-3
        nx = int(round(x_span / self.dx)); nx += nx % 2
        self.nx = nx
        self.x = geo.xc + (np.arange(nx) - (nx - 1) / 2) * self.dx
        # snap elements to the nearest grid node
        self.el_idx = np.array([np.argmin(np.abs(self.x - e)) for e in geo.elx])
        # finite element width: each element covers `el_nodes` grid nodes (directivity,
        # suppresses grating lobes like a real / k-Wave element)
        w = max(1, int(round(el_width_frac * pitch / self.dx)))
        off = np.arange(w) - (w - 1) // 2
        self.el_nodes = np.clip(self.el_idx[:, None] + off[None, :], 0, nx - 1)     # (nel, w)
        self.z_max = z_max or (geo.nz * geo.dz + 1.5e-3)
        self.nz = int(np.ceil(self.z_max / self.dz))
        self.z = np.arange(self.nz) * self.dz
        self.fs_out = fs_out
        t_max = 2.2 * self.z_max / 1380 + 4e-6
        self.nt = int(np.ceil(t_max * fs_out / 16) * 16)
        self.T = self.nt / fs_out
        df = 1 / self.T
        sig = frac_bw * fc / 2.355                                        # Gaussian pulse spectrum
        f = np.arange(int((fc - 2.5 * sig) / df), int((fc + 2.5 * sig) / df) + 1) * df
        self.f = f[f > 0.5e6]
        self.P = np.exp(-0.5 * ((self.f - fc) / sig) ** 2).astype(np.float32)
        self.f_chunk = f_chunk or 48
        xmax = (np.abs(geo.elx - geo.xc).max() + np.abs(self.x - geo.xc).max()) / 2
        self.aawin = (1 / np.sqrt(1 + ((self.x - geo.xc) / xmax) ** ord_aa)).astype(np.float32)
        self.t_out = np.arange(self.nt) / fs_out


def simulate_pw_iq(B, cfg: SimConfig, sos, refl):
    """sos, refl: (cfg.nz, cfg.nx) arrays (refl complex). Returns baseband IQ
    channel data (nang, nel, nt) and t_out — same format as acq.fsa_to_pw_iq."""
    geo = cfg.geo
    nang = geo.n_tx; nel = len(geo.elx)
    kx = B.asarray(np.fft.fftfreq(cfg.nx, cfg.dx).astype(np.float32))
    s = B.asarray((1.0 / sos).astype(np.float32))                          # (nz, nx)
    smean = B.mean(s, axis=1, keepdims=True)                               # (nz, 1)
    ds = s - smean
    refl = B.asarray(refl.astype(np.complex64))
    W = B.asarray(cfg.aawin)
    el_idx = cfg.el_idx
    # transmit delays / apodisation (programmed with c_bf)
    tau = geo.tx_delay.astype(np.float32); apod = geo.tx_apod.astype(np.float32)
    Y = []                                                                 # per freq chunk: (nang, nfc, nel)
    fchunk = cfg.f_chunk if B.name == 'numpy' else max(cfg.f_chunk, 128)      # bigger chunks on GPU
    for f0 in range(0, len(cfg.f), fchunk):
        fnp = cfg.f[f0:f0 + fchunk]
        f = B.asarray(fnp.astype(np.float32))
        nf = len(fnp)
        # boundary transmit field at z=0: delta at each element with programmed delay
        T0 = np.zeros((nang, nf, cfg.nx), np.complex64)
        ph = np.exp(-2j * np.pi * fnp[None, :, None] * tau[:, None, :]) * cfg.P[f0:f0 + nf][None, :, None] * apod[:, None, :]
        for j in range(cfg.el_nodes.shape[1]):
            T0[:, :, cfg.el_nodes[:, j]] += ph / cfg.el_nodes.shape[1]
        Tz = B.asarray(T0)
        # store source terms T(z) * r(z) for every depth
        srcs = []
        for iz in range(cfg.nz):
            srcs.append(Tz * refl[iz][None, None, :])
            if iz < cfg.nz - 1:
                Tz = _step(B, Tz, f, kx, smean[iz], ds[iz], cfg.dz, W)
        U = B.zeros((nang, nf, cfg.nx), B.cfloat)
        for iz in range(cfg.nz - 1, -1, -1):
            U = U + srcs[iz]
            if iz > 0:
                U = _step(B, U, f, kx, smean[iz - 1], ds[iz - 1], cfg.dz, W)
        del srcs
        Yc = 0
        for j in range(cfg.el_nodes.shape[1]):
            Yc = Yc + U[:, :, B.asarray(cfg.el_nodes[:, j])]
        Y.append(Yc / cfg.el_nodes.shape[1])                               # (nang, nf, nel)
    Y = B.cat(Y, axis=1)
    # baseband synthesis  iq(t) = sum_f Y(f) exp(i2pi f t) exp(-i2pi fc t)
    E = np.exp(2j * np.pi * (cfg.f[:, None] - cfg.fc) * cfg.t_out[None, :]).astype(np.complex64)
    E = B.asarray(E)
    iq = B.einsum('afe,fn->aen', Y, E)
    return iq, cfg.t_out


def _step(B, F, f, kx, smean, ds, dz, W):
    """One split-step of the field F (nang, nf, nx) through a layer of thickness dz."""
    kz2 = (f[:, None] * smean) ** 2 - kx[None, :] ** 2                     # (nf, nx)
    prop = (kz2 > 0)
    kz = B.sqrt(B.clip(kz2, 0, None))
    H = B.polar(-2 * np.pi * kz * dz) * (prop.astype(np.float32) if B.name == 'numpy' else prop.float())
    S = B.polar(-2 * np.pi * f[:, None] * ds[None, :] * dz)                 # (nf, nx)
    return S[None] * W * B.ifft(H[None] * B.fft(W * F, axis=-1), axis=-1)


def make_medium(rng, cfg: SimConfig, sos_img, img_extent, echo=None, speckle_density=1.0):
    """Interpolate an SoS map defined on the recon field of view (img_extent =
    (z0,z1,x0,x1) [m]) to the simulation grid (edge-extended), and draw a random
    complex reflectivity (speckle x echogenicity)."""
    from scipy.ndimage import map_coordinates
    nzi, nxi = sos_img.shape
    z0, z1, x0, x1 = img_extent
    fz = (cfg.z - z0) / (z1 - z0) * nzi - 0.5
    fx = (cfg.x - x0) / (x1 - x0) * nxi - 0.5
    ZZ, XX = np.meshgrid(fz, fx, indexing='ij')
    sos = map_coordinates(sos_img, [ZZ, XX], order=1, mode='nearest').astype(np.float32)
    amp = np.ones_like(sos) if echo is None else map_coordinates(echo, [ZZ, XX], order=1, mode='nearest')
    r = (rng.standard_normal(sos.shape) + 1j * rng.standard_normal(sos.shape)) * amp * speckle_density
    r[cfg.z < 0.5e-3] = 0                                                  # no scatterers at the lens
    return sos, r.astype(np.complex64)

"""Tiny array-backend shim so the physics kernels run on NumPy (CPU, testing)
or PyTorch (GPU, autograd).  Only the operations used in this package are wrapped.

Usage:
    B = get_backend('torch', device='cuda:0')   # or get_backend('numpy')
    x = B.asarray(np_array)
"""
import numpy as np


class NumpyBackend:
    name = 'numpy'
    device = 'cpu'
    cfloat = np.complex64
    float = np.float32
    pi = np.pi

    def asarray(self, x, dtype=None):
        if hasattr(x, 'detach'):                      # torch tensor -> numpy
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or np.float32)

    def arange(self, *a, dtype=None):
        return np.arange(*a, dtype=dtype or np.float32)

    def linspace(self, a, b, n):
        return np.linspace(a, b, n, dtype=np.float32)

    # elementwise
    exp = staticmethod(np.exp); sqrt = staticmethod(np.sqrt); abs = staticmethod(np.abs)
    angle = staticmethod(np.angle); conj = staticmethod(np.conj); real = staticmethod(np.real)
    imag = staticmethod(np.imag); sin = staticmethod(np.sin); cos = staticmethod(np.cos)
    tan = staticmethod(np.tan); floor = staticmethod(np.floor); log = staticmethod(np.log)
    log1p = staticmethod(np.log1p); maximum = staticmethod(np.maximum); minimum = staticmethod(np.minimum)
    where = staticmethod(np.where); sign = staticmethod(np.sign); atan2 = staticmethod(np.arctan2)

    def clip(self, x, lo=None, hi=None):
        return np.clip(x, lo, hi)

    def csqrt(self, x):
        return np.sqrt(x.astype(np.complex64))

    def polar(self, phase):
        return np.exp(1j * phase).astype(np.complex64)

    def complex(self, re, im):
        return (re + 1j * im).astype(np.complex64)

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def cumsum(self, x, axis):
        return np.cumsum(x, axis=axis)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def cat(self, xs, axis=0):
        return np.concatenate(xs, axis=axis)

    def einsum(self, eq, *ops):
        return np.einsum(eq, *ops, optimize=True)

    def fft(self, x, axis=-1):
        return np.fft.fft(x, axis=axis).astype(np.complex64)

    def ifft(self, x, axis=-1):
        return np.fft.ifft(x, axis=axis).astype(np.complex64)

    def fftfreq(self, n, d):
        return np.fft.fftfreq(n, d).astype(np.float32)

    def take_along(self, x, idx, axis):
        return np.take_along_axis(x, idx, axis=axis)

    def pad(self, x, pads):           # pads: list of (before, after) per axis
        return np.pad(x, pads)

    def spmm(self, A, x):             # scipy sparse @ dense
        return A @ x

    def spmm_T(self, A, x):
        return A.T @ x

    def sparse(self, A_scipy):
        return A_scipy.tocsr().astype(np.float32)

    def interp2(self, img, zq, xq):
        """Bilinear sample img[Nz,Nx] at fractional index coords (zq, xq) (any shape),
        clamped at the border."""
        Nz, Nx = img.shape
        zq = np.clip(zq, 0, Nz - 1); xq = np.clip(xq, 0, Nx - 1)
        z0 = np.clip(np.floor(zq).astype(np.int64), 0, Nz - 2); x0 = np.clip(np.floor(xq).astype(np.int64), 0, Nx - 2)
        wz = zq - z0; wx = xq - x0
        return ((1 - wz) * (1 - wx) * img[z0, x0] + (1 - wz) * wx * img[z0, x0 + 1]
                + wz * (1 - wx) * img[z0 + 1, x0] + wz * wx * img[z0 + 1, x0 + 1])

    def median3(self, x):
        from scipy.ndimage import median_filter
        return median_filter(x, size=3, mode='nearest')

    def no_grad(self):
        import contextlib
        return contextlib.nullcontext()


class TorchBackend:
    name = 'torch'

    def __init__(self, device='cuda'):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.cfloat = torch.complex64
        self.float = torch.float32
        self.pi = float(np.pi)

    def _dt(self, dtype):
        t = self.torch
        if dtype is None: return None
        m = {np.float32: t.float32, np.float64: t.float64, np.complex64: t.complex64,
             np.int64: t.int64, np.bool_: t.bool, float: t.float32, complex: t.complex64, bool: t.bool}
        if isinstance(dtype, t.dtype): return dtype
        return m.get(dtype, m.get(np.dtype(dtype).type))

    def asarray(self, x, dtype=None):
        t = self.torch
        if isinstance(x, t.Tensor):
            x = x.to(self.device)
            return x.to(self._dt(dtype)) if dtype is not None else x
        x = np.asarray(x)
        if dtype is None:
            if x.dtype == np.float64: dtype = np.float32
            if x.dtype == np.complex128: dtype = np.complex64
        if dtype is not None: x = x.astype(dtype)
        return t.from_numpy(np.ascontiguousarray(x)).to(self.device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(shape, dtype=self._dt(dtype) or self.torch.float32, device=self.device)

    def ones(self, shape, dtype=None):
        return self.torch.ones(shape, dtype=self._dt(dtype) or self.torch.float32, device=self.device)

    def arange(self, *a, dtype=None):
        return self.torch.arange(*a, dtype=self._dt(dtype) or self.torch.float32, device=self.device)

    def linspace(self, a, b, n):
        return self.torch.linspace(a, b, n, device=self.device)

    def __getattr__(self, name):
        t = self.__dict__['torch']
        table = dict(exp=t.exp, sqrt=t.sqrt, abs=t.abs, angle=t.angle, conj=t.conj, real=t.real,
                     imag=t.imag, sin=t.sin, cos=t.cos, tan=t.tan, floor=t.floor, log=t.log,
                     log1p=t.log1p, maximum=t.maximum, minimum=t.minimum, where=t.where,
                     sign=t.sign, atan2=t.atan2)
        if name in table: return table[name]
        raise AttributeError(name)

    def clip(self, x, lo=None, hi=None):
        return self.torch.clamp(x, lo, hi)

    def csqrt(self, x):
        return self.torch.sqrt(x.to(self.torch.complex64))

    def polar(self, phase):
        return self.torch.polar(self.torch.ones_like(phase), phase)

    def complex(self, re, im):
        return self.torch.complex(re, im)

    def sum(self, x, axis=None, keepdims=False):
        return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return x.mean() if axis is None else x.mean(dim=axis, keepdim=keepdims)

    def cumsum(self, x, axis):
        return self.torch.cumsum(x, dim=axis)

    def stack(self, xs, axis=0):
        return self.torch.stack(xs, dim=axis)

    def cat(self, xs, axis=0):
        return self.torch.cat(xs, dim=axis)

    def einsum(self, eq, *ops):
        return self.torch.einsum(eq, *ops)

    def fft(self, x, axis=-1):
        return self.torch.fft.fft(x, dim=axis)

    def ifft(self, x, axis=-1):
        return self.torch.fft.ifft(x, dim=axis)

    def fftfreq(self, n, d):
        return self.torch.fft.fftfreq(n, d, device=self.device)

    def take_along(self, x, idx, axis):
        return self.torch.take_along_dim(x, idx, dim=axis)

    def pad(self, x, pads):
        flat = []
        for b, a in reversed(pads): flat += [b, a]
        return self.torch.nn.functional.pad(x, flat)

    def sparse(self, A_scipy):
        A = A_scipy.tocoo()
        i = np.vstack([A.row, A.col]).astype(np.int64)
        T = self.torch.sparse_coo_tensor(self.torch.from_numpy(i), self.torch.from_numpy(A.data.astype(np.float32)),
                                         A.shape).coalesce().to(self.device)
        TT = self.torch.sparse_coo_tensor(self.torch.from_numpy(i[::-1].copy()),
                                          self.torch.from_numpy(A.data.astype(np.float32)),
                                          (A.shape[1], A.shape[0])).coalesce().to(self.device)
        return (T.to_sparse_csr(), TT.to_sparse_csr())

    def spmm(self, A, x):
        squeeze = x.dim() == 1
        y = A[0] @ (x[:, None] if squeeze else x.contiguous())
        return y[:, 0] if squeeze else y

    def spmm_T(self, A, x):
        squeeze = x.dim() == 1
        y = A[1] @ (x[:, None] if squeeze else x.contiguous())
        return y[:, 0] if squeeze else y

    def interp2(self, img, zq, xq):
        """Differentiable bilinear sampling of img[Nz,Nx] at fractional index coords."""
        F = self.torch.nn.functional
        Nz, Nx = img.shape
        gx = 2 * xq / max(Nx - 1, 1) - 1
        gz = 2 * zq / max(Nz - 1, 1) - 1
        shp = xq.shape
        grid = self.torch.stack([gx.reshape(1, -1, 1), gz.reshape(1, -1, 1)], dim=-1)
        out = F.grid_sample(img[None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out.reshape(shp)

    def median3(self, x):
        F = self.torch.nn.functional
        p = F.pad(x[None, None], (1, 1, 1, 1), mode='replicate')
        u = p.unfold(2, 3, 1).unfold(3, 3, 1)
        return u.reshape(*u.shape[:4], 9).median(dim=-1).values[0, 0]

    def no_grad(self):
        return self.torch.no_grad()


def get_backend(name='numpy', device='cuda'):
    if name == 'numpy':
        return NumpyBackend()
    return TorchBackend(device)

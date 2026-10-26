"""Analytical reconstruction: L1-TV / L2-TV via Chambolle-Pock (batched), and the
conjugate-gradient data-consistency step used inside DenOiS' plug-and-play loop.

All solvers work in *normalised units*:
    A_n = A / s1         (s1 = largest singular value of A -> "SVD-normalised")
    u_n = u / U_SCALE    (u in ns/mm, U_SCALE ~ typical magnitude)
    b_n = b / (U_SCALE * s1)
Arrays: u is (nb, nz, nx); b and w (weights/masks) are (nb, M).
"""
import numpy as np

U_SCALE = 30.0      # ns/mm  (|1/1400-1/1540|*1e6 = 65, |1/1650-1/1540|*1e6 = 43)


def largest_singular_value(A, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1]).astype(np.float64)
    for _ in range(iters):
        v = A.T @ (A @ v); v /= np.linalg.norm(v)
    return float(np.sqrt(np.linalg.norm(A.T @ (A @ v))))


class LinearOp:
    """Holds the (normalised) operator on a backend, with batched apply/adjoint."""
    def __init__(self, B, A_scipy, nz, nx, s1=None):
        self.B = B
        self.s1 = s1 or largest_singular_value(A_scipy)
        self.A = B.sparse((A_scipy / self.s1).astype(np.float32))
        self.nz, self.nx = nz, nx
        self.M = A_scipy.shape[0]
        An = abs(A_scipy) / self.s1
        # diagonal preconditioners (Pock & Chambolle 2011, alpha=1)
        rs = np.asarray(An.sum(1)).ravel(); cs = np.asarray(An.sum(0)).ravel()
        self.sig_A = B.asarray((1.0 / np.maximum(rs, 1e-6)).astype(np.float32))[None, :]
        self.tau = B.asarray((1.0 / (cs + 4.0)).astype(np.float32)).reshape(1, nz, nx)

    def fwd(self, u):                       # (nb, nz, nx) -> (nb, M)
        nb = u.shape[0]
        return self.B.spmm(self.A, u.reshape(nb, -1).T).T

    def adj(self, y):                       # (nb, M) -> (nb, nz, nx)
        nb = y.shape[0]
        return self.B.spmm_T(self.A, y.T).T.reshape(nb, self.nz, self.nx)

    # unit conversions
    def b_to_n(self, b_ns):  return b_ns / (U_SCALE * self.s1)
    def n_to_b(self, b_n):   return b_n * (U_SCALE * self.s1)


def grad(B, u):
    gz = B.cat([u[:, 1:] - u[:, :-1], 0 * u[:, :1]], axis=1)
    gx = B.cat([u[:, :, 1:] - u[:, :, :-1], 0 * u[:, :, :1]], axis=2)
    return gz, gx


def div(B, pz, px):     # = -grad^T
    dz = B.cat([pz[:, :1], pz[:, 1:-1] - pz[:, :-2], -pz[:, -2:-1]], axis=1)
    dx = B.cat([px[:, :, :1], px[:, :, 1:-1] - px[:, :, :-2], -px[:, :, -2:-1]], axis=2)
    return dz + dx


def tv_recon(op: LinearOp, b_n, w, lam=0.1, p=2, iters=300, u0=None):
    """min_u  data_p(w*(A u - b)) + lam_eff * TV_iso(u)   (Chambolle-Pock / PDHG)
       p=2: 0.5*||.||^2,  p=1: ||.||_1.   w in [0,1] (0 = missing measurement).
    `lam` is a *relative* weight: lam_eff = lam * ||A^T g||_inf per sample, with
    g = w*b (p=2) or w*sign(b) (p=1) - the data-term gradient at u=0, so the same
    `lam` behaves consistently across datasets/scales."""
    B = op.B
    nb = b_n.shape[0]
    g = w * b_n if p == 2 else w * B.sign(b_n)
    lam = lam * B.abs(op.adj(g)).reshape(nb, -1).max(1)[:, None, None] if B.name == 'numpy' \
        else lam * B.abs(op.adj(g)).reshape(nb, -1).amax(1)[:, None, None]
    u = B.zeros((nb, op.nz, op.nx)) if u0 is None else u0
    ubar = u
    y = B.zeros((nb, op.M)); qz = B.zeros((nb, op.nz, op.nx)); qx = B.zeros((nb, op.nz, op.nx))
    sig = op.sig_A; tau = op.tau; sig_g = 0.5
    for _ in range(iters):
        # dual data (diagonally preconditioned)
        y = y + sig * op.fwd(ubar)
        if p == 2:
            y = B.where(w > 0, (y - sig * b_n) / (1 + sig / B.clip(w, 1e-6, None)), 0 * y)
        else:
            v = y - sig * b_n
            y = B.minimum(B.maximum(v, -w), w)
        # dual TV
        gz, gx = grad(B, ubar)
        qz = qz + sig_g * gz; qx = qx + sig_g * gx
        nrm = B.maximum(B.sqrt(qz ** 2 + qx ** 2) / lam, 1.0 + 0 * qz)
        qz = qz / nrm; qx = qx / nrm
        # primal
        u_old = u
        u = u - tau * (op.adj(y) - div(B, qz, qx))
        ubar = 2 * u - u_old
    return u


def dc_cg(op: LinearOp, b_n, w, z, nu, iters=10):
    """Data-consistency step (3b):  argmin_x ||w*(A x - b)||^2 + nu ||x - z||^2
    solved with batched conjugate gradient, warm-started at z."""
    B = op.B
    def Hx(x): return op.adj(w * op.fwd(x)) + nu * x
    x = z
    r = op.adj(w * b_n) + nu * z - Hx(x)
    pdir = r
    rs = B.sum((r * r).reshape(r.shape[0], -1), axis=1)
    for _ in range(iters):
        Ap = Hx(pdir)
        alpha = rs / (B.sum((pdir * Ap).reshape(r.shape[0], -1), axis=1) + 1e-12)
        x = x + alpha[:, None, None] * pdir
        r = r - alpha[:, None, None] * Ap
        rs_new = B.sum((r * r).reshape(r.shape[0], -1), axis=1)
        pdir = r + (rs_new / (rs + 1e-12))[:, None, None] * pdir
        rs = rs_new
    return x


def tv_irls(op: LinearOp, b_n, w, lam=1e-3, p=2, outer=12, inner=25, eps=1e-3, delta=None, u0=None):
    """min_u  data_p(w*(A u - b)) + lam * sum sqrt(|grad u|^2 + eps^2)
    by iteratively re-weighted least squares; each outer step solves
        (A^T Om A + lam G^T D G) u = A^T Om b      with batched CG (warm start)
    Om = w (p=2) or w / max(|r|, delta) (p=1);  D = 1 / sqrt(|grad u|^2 + eps^2).
    Converges robustly for both norms (unlike plain PDHG on this ill-posed operator)."""
    B = op.B
    nb = b_n.shape[0]
    u = B.zeros((nb, op.nz, op.nx)) if u0 is None else u0
    if delta is None:
        delta = 0.05 * B.mean(B.abs(b_n), axis=1, keepdims=True) + 1e-9
    Om = w
    for it in range(outer):
        gz, gx = grad(B, u)
        D = 1.0 / B.sqrt(gz ** 2 + gx ** 2 + eps ** 2)
        if p == 1 and it > 0:
            r = op.fwd(u) - b_n
            Om = w / B.maximum(B.abs(r), delta + 0 * r)
            Om = Om / (B.mean(Om, axis=1, keepdims=True) / (B.mean(w, axis=1, keepdims=True) + 1e-9) + 1e-12)

        def H(v):
            vz, vx = grad(B, v)
            return op.adj(Om * op.fwd(v)) - lam * div(B, D * vz, D * vx)
        rhs = op.adj(Om * b_n)
        x = u
        r = rhs - H(x); pdir = r
        rs = B.sum((r * r).reshape(nb, -1), axis=1)
        for _ in range(inner):
            Ap = H(pdir)
            alpha = rs / (B.sum((pdir * Ap).reshape(nb, -1), axis=1) + 1e-20)
            x = x + alpha[:, None, None] * pdir
            r = r - alpha[:, None, None] * Ap
            rs_new = B.sum((r * r).reshape(nb, -1), axis=1)
            pdir = r + (rs_new / (rs + 1e-20))[:, None, None] * pdir
            rs = rs_new
        u = x
    return u

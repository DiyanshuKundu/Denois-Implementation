"""DenOiS inference: observation refinement (M_U / M_D) + diffusion plug-and-play
reconstruction (R_D) with conjugate-gradient data consistency, plus the baselines
(L1-TV, L2-TV, direct Unet [Chen&Han], DOLCE, Unet(M_U)) evaluated on the same inputs.
"""
import os, time
import numpy as np
import torch

from .backend import get_backend
from .datagen import Ops
from .nets import sos_norm, sos_denorm, B_SCALE
from .raymodel import sos_to_u, u_to_sos
from .solvers import dc_cg, U_SCALE
from .train import load_model, refiner_inputs
from .metrics import all_metrics


SIM_PNP = dict(t0=15, K=5)       # paper: in-silico T0=15, K=5
REAL_PNP = dict(t0=50, K=10)     # paper: real data  T0=50, K=10


class DenoisRunner:
    def __init__(self, ckpt_dir, geo, device='cuda', nu=0.03, cg_iters=5):
        self.dev = device
        self.B = get_backend('torch', device)
        self.geo = geo
        self.ops = Ops(self.B, geo, with_hi=False)
        self.models = {}
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith('.pt') and not f.endswith('_S.pt'):
                self.models[f[:-3]] = load_model(os.path.join(ckpt_dir, f), device)
        print('loaded models:', sorted(self.models))
        self.nu, self.cg_iters = nu, cg_iters
        Z = np.meshgrid(geo.z, geo.x, indexing='ij')[0]
        self.depth_ok = np.broadcast_to((Z >= geo.zmin_valid)[None], (geo.npair, geo.nz, geo.nx)).astype(np.float32)

    # ------------------------------------------------------------- observation domain
    @torch.no_grad()
    def refine(self, b, w, xprime, name, seed=0):
        """b, w: (n,8,64,64); xprime: (n,64,64) SoS.  name e.g. 'mu_bPA' / 'md_bP'.
        Returns refined delays (n,8,64,64) [ns] and weights (inpainted -> all valid)."""
        variant = name.split('_')[1]
        X = torch.from_numpy(refiner_inputs(b, w, xprime, variant, self.geo)).to(self.dev)
        model, diff = self.models[name]
        outs = []
        for s in range(0, len(X), 128):
            c = X[s:s + 128]
            if diff is None:
                outs.append(model(c))
            else:
                outs.append(diff.ddpm(model, c, (len(c), 1, *c.shape[2:]), seed=seed + s))
        y = torch.cat(outs).cpu().numpy()[:, 0] * B_SCALE
        n = b.shape[0]
        wr = np.broadcast_to(self.depth_ok[None], b.shape).astype(np.float32)
        return y.reshape(b.shape).astype(np.float32), wr

    def tv(self, b, w):
        return self.ops.tv(b, w, 'l1'), self.ops.tv(b, w, 'l2')

    # ------------------------------------------------------------- solution domain
    def _dc_fn(self, b, w, nu=None):
        B, op = self.B, self.ops.op
        n = b.shape[0]
        bn = B.asarray(op.b_to_n(b.reshape(n, -1).astype(np.float32)))
        ww = B.asarray(w.reshape(n, -1).astype(np.float32))
        nu = self.nu if nu is None else nu
        cbf = self.geo.c_bf

        def dc(x0):
            c = sos_denorm(x0[:, 0]).clamp(1300, 1750)
            u = (1.0 / c - 1.0 / cbf) * 1e6 / U_SCALE
            u = dc_cg(op, bn * ww, ww, u, nu, self.cg_iters)
            c2 = 1.0 / (1.0 / cbf + u * U_SCALE * 1e-6)
            return sos_norm(c2)[:, None]
        return dc

    @torch.no_grad()
    def rd(self, b, w, l1, l2, real=False, nu=None, seed=0, model='rd', guidance=0.0):
        """DenOiS reconstruction module R_D: start from the L1-TV proposal + noise at T0,
        DDIM denoising conditioned on {L1-TV, L2-TV}, CG data consistency every K steps."""
        m, diff = self.models[model]
        p = REAL_PNP if real else SIM_PNP
        cond = torch.from_numpy(np.stack([sos_norm(l1), sos_norm(l2)], 1).astype(np.float32)).to(self.dev)
        x_init = cond[:, :1].clone()
        x0 = diff.ddim(m, cond, x_init=x_init, t0=p['t0'], dc=self._dc_fn(b, w, nu), K=p['K'],
                       guidance=guidance, seed=seed)
        return sos_denorm(x0[:, 0]).cpu().numpy()

    @torch.no_grad()
    def dolce(self, b, w, l2, nu=None, seed=0):
        """DOLCE: conditioned on L2-TV only, from pure noise, DC after every step."""
        m, diff = self.models['dolce']
        cond = torch.from_numpy(sos_norm(l2)[:, None].astype(np.float32)).to(self.dev)
        x0 = diff.ddpm(m, cond, cond.shape, dc=self._dc_fn(b, w, nu), K=1, seed=seed)
        return sos_denorm(x0[:, 0]).cpu().numpy()

    @torch.no_grad()
    def unetR(self, l1, l2):
        m, _ = self.models['unetR']
        c = torch.from_numpy(np.stack([sos_norm(l1), sos_norm(l2)], 1).astype(np.float32)).to(self.dev)
        return sos_denorm(m(c)[:, 0]).cpu().numpy()

    @torch.no_grad()
    def direct(self, b, w):
        m, _ = self.models['direct']
        c = torch.from_numpy(np.concatenate([b / B_SCALE, w], 1).astype(np.float32)).to(self.dev)
        return sos_denorm(m(c)[:, 0]).cpu().numpy()

    # ------------------------------------------------------------- full grid of methods
    def run_grid(self, cases, real=False, refiners=('mu_b', 'mu_bP', 'mu_bPA', 'md_bPA'),
                 recons=('L2-TV', 'L1-TV', 'Unet[5]', 'Dolce', 'Unet(MU)', 'RD'), n_outer=1, log=print):
        """cases: dict with stacked arrays b, w, l1, l2 (n,...).  Returns
        {input_name: {recon_name: (n,64,64) SoS}} and the refined observations."""
        b, w, l1, l2 = cases['b'], cases['w'], cases['l1'], cases['l2']
        out, obs = {}, {}
        inputs = [("b'", b, w, l1, l2)]
        for r in refiners:
            if r not in self.models:
                continue
            t0 = time.time()
            xprime = l1
            for it in range(n_outer):
                br, wr = self.refine(b, w, xprime, r)
                r1, r2 = self.tv(br, wr)
                if n_outer > 1 and 'rd' in self.models:
                    xprime = self.rd(br, wr, r1, r2, real=real)
            name = {'mu': 'MU', 'md': 'MD'}[r[:2]] + '(' + {'b': "b'", 'bP': "b',P", 'bPA': "b',P,A'x'"}[r.split('_')[1]] + ')'
            inputs.append((name, br, wr, r1, r2)); obs[name] = br
            log(f'  refined with {name} in {time.time() - t0:.1f}s')
        for name, bb, ww, r1, r2 in inputs:
            res = {}
            for rc in recons:
                t0 = time.time()
                if rc == 'L2-TV': res[rc] = r2
                elif rc == 'L1-TV': res[rc] = r1
                elif rc == 'Unet[5]' and 'direct' in self.models and name == "b'": res[rc] = self.direct(bb, ww)
                elif rc == 'Dolce' and 'dolce' in self.models: res[rc] = self.dolce(bb, ww, r2)
                elif rc == 'Unet(MU)' and 'unetR' in self.models: res[rc] = self.unetR(r1, r2)
                elif rc == 'RD' and 'rd' in self.models: res[rc] = self.rd(bb, ww, r1, r2, real=real)
            out[name] = res
            log(f'  {name}: ' + ', '.join(res))
        return out, obs


def score_grid(grid, gt, geo):
    """Mean metrics per (input, recon) over cases with ground truth."""
    table = {}
    for inp, rs in grid.items():
        for rc, X in rs.items():
            ms = [all_metrics(X[i], gt[i], geo) for i in range(len(gt))]
            keys = set.intersection(*[set(m) for m in ms])
            table[(inp, rc)] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in keys}
    return table


def print_table(table, metric='RMSE', second='ACE'):
    inps = list(dict.fromkeys(k[0] for k in table)); rcs = list(dict.fromkeys(k[1] for k in table))
    hdr = f"{'Input':<16}" + ''.join(f'{r:>18}' for r in rcs)
    lines = [hdr, '-' * len(hdr)]
    for i in inps:
        row = f'{i:<16}'
        for r in rcs:
            v = table.get((i, r))
            if v is None: row += f"{'':>18}"
            else:
                a = v[metric][0]; s = f'{a:.2f}'
                if second in v: s += f' ({v[second][0]:.2f})'
                row += f'{s:>18}'
        lines.append(row)
    return '\n'.join(lines)

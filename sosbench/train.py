"""Training CLI for every learned component.

    python -m sosbench.train --model rd     --data S.npz  --epochs 150 --out ckpt/rd_S.pt
    python -m sosbench.train --model rd     --data C1.npz --epochs 100 --init ckpt/rd_S.pt --out ckpt/rd.pt
    python -m sosbench.train --model mu_bPA --data C1.npz --epochs 150 --out ckpt/mu_bPA.pt

Models
    rd       DenOiS reconstruction diffusion R_D, conditioned on {L1-TV, L2-TV}       (eps, L2)
    dolce    DOLCE baseline diffusion, conditioned on L2-TV                           (eps, L2)
    unetR    Unet(M_U): U-Net {L1-TV, L2-TV} -> x  (simple R inside DenOiS)            (L2)
    direct   Unet [Chen & Han]: delay maps + masks -> x                                (L2)
    mu_<v>   U-Net measurement refiner M_U,  v in {b, bP, bPA}                         (L1)
    md_<v>   diffusion measurement refiner M_D, v in {b, bP, bPA}                      (eps, L2)
"""
import argparse, copy, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .nets import UNet, Diffusion, sos_norm, B_SCALE, count_params
from .acq import default_geometry
from .raymodel import build_operator, aperture_mask, directivity_maps, sos_to_u

COND_CH = dict(b=1, bP=5, bPA=6)


def build(kind):
    """Returns (model, diffusion-or-None, description)."""
    if kind in ('rd', 'dolce'):
        nc = 2 if kind == 'rd' else 1
        return UNet(1 + nc + 1, 1, base=64, mults=(1, 2, 2, 2), use_t=True), Diffusion(200), f'diffusion, {nc} cond'
    if kind == 'unetR':
        return UNet(2, 1, base=64, use_t=False, dropout=0.0), None, 'unet l1,l2->x'
    if kind == 'direct':
        return UNet(16, 1, base=64, use_t=False, dropout=0.0), None, 'unet b,w->x'
    v = kind.split('_')[1]
    if kind.startswith('mu_'):
        return UNet(COND_CH[v], 1, base=64, mults=(1, 2, 2, 2), attn_levels=(), use_t=False, dropout=0.0), None, 'refiner unet'
    if kind.startswith('md_'):
        return UNet(1 + COND_CH[v] + 1, 1, base=64, mults=(1, 2, 2, 2), use_t=True), Diffusion(200), 'refiner diffusion'
    raise ValueError(kind)


# ------------------------------------------------------------------ refiner inputs
_GEO_CACHE = {}


def refiner_static(geo=None):
    """Static prior channels P (except the data-dependent low-correlation mask)."""
    key = 'default' if geo is None else id(geo)
    geo = geo or default_geometry()
    if key not in _GEO_CACHE:
        _GEO_CACHE[key] = dict(ap=aperture_mask(geo).astype(np.float32), dirm=directivity_maps(geo),
                               A=build_operator(geo))
    return _GEO_CACHE[key]


def refiner_inputs(b, w, xprime, variant, geo=None):
    """Per-pair refiner input stack.
    b, w: (n, 8, 64, 64) observed delays [ns] and validity; xprime: (n, 64, 64) SoS
    estimate used for the A'x' prior.  Returns float32 (n*8, C, 64, 64)."""
    st = refiner_static(geo)
    n, npair = b.shape[:2]
    chans = [b / B_SCALE]
    if 'P' in variant:
        ap = np.broadcast_to(st['ap'][None], b.shape)
        lowc = ap * (1 - w)                                                   # in aperture but below SNR
        chans += [ap, lowc, np.broadcast_to(st['dirm'][None, :, 0], b.shape), np.broadcast_to(st['dirm'][None, :, 1], b.shape)]
    if 'A' in variant:
        Ax = (st['A'] @ sos_to_u(xprime.reshape(n, -1)).T).T.reshape(b.shape)
        chans.append(Ax / B_SCALE)
    X = np.stack(chans, 2).astype(np.float32)                               # (n, 8, C, 64, 64)
    return X.reshape(n * npair, X.shape[2], *b.shape[2:])


# ------------------------------------------------------------------ data
def load_xy(kind, path):
    d = np.load(path)
    x = sos_norm(d['x'].astype(np.float32))[:, None]                        # (n,1,64,64)
    if kind == 'rd':
        return np.concatenate([sos_norm(d['l1'])[:, None], sos_norm(d['l2'])[:, None]], 1).astype(np.float32), x
    if kind == 'dolce':
        return sos_norm(d['l2'])[:, None].astype(np.float32), x
    if kind == 'unetR':
        return np.concatenate([sos_norm(d['l1'])[:, None], sos_norm(d['l2'])[:, None]], 1).astype(np.float32), x
    if kind == 'direct':
        b = d['b'].astype(np.float32) / B_SCALE; w = d['w'].astype(np.float32)
        return np.concatenate([b, w], 1), x
    v = kind.split('_')[1]
    cond = refiner_inputs(d['b'].astype(np.float32), d['w'].astype(np.float32), d['l1'].astype(np.float32), v)
    y = (d['Ax'].astype(np.float32) / B_SCALE).reshape(-1, 1, *d['Ax'].shape[2:])
    return cond, y


# ------------------------------------------------------------------ training
def train(kind, data, epochs, out, init=None, bs=64, lr=None, lr_end=None, val_frac=0.05, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, diff, desc = build(kind)
    if init:
        model.load_state_dict(torch.load(init, map_location='cpu')['model'])
    model = model.to(dev)
    if diff is not None: diff.to(dev)
    ema = copy.deepcopy(model).eval()
    cond, y = load_xy(kind, data)
    n = len(y); nv = max(1, int(val_frac * n))
    perm = np.random.permutation(n); vi, ti = perm[:nv], perm[nv:]
    C = torch.from_numpy(cond); Y = torch.from_numpy(y)
    is_ref = kind.startswith('mu_') or kind.startswith('md_')
    lr = lr or (1e-4 if is_ref else 1e-3)
    lr_end = lr_end or lr / 100
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * max(1, len(ti) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (lr_end / lr) ** min(1.0, s / max(1, steps)))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    log(f'[{kind}] {desc}  params={count_params(model):.1f}M  n_train={len(ti)} n_val={nv}  epochs={epochs}')
    t0 = time.time(); best = 1e9
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(ti)
        tl = 0.0; nbat = 0
        for s in range(0, len(idx) - bs + 1 if len(idx) >= bs else 1, bs):
            j = torch.from_numpy(idx[s:s + bs])
            c = C[j].to(dev, non_blocking=True); yy = Y[j].to(dev, non_blocking=True)
            c, yy = _augment(c, yy, kind)
            with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                if diff is not None:
                    loss = diff.loss(model, yy, c)
                else:
                    pred = model(c)
                    loss = F.l1_loss(pred, yy) if is_ref else F.mse_loss(pred, yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            gstep = ep * max(1, len(ti) // bs) + nbat
            decay = min(0.999, (1 + gstep) / (10 + gstep))           # EMA warm-up (short fine-tunes)
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            tl += loss.item(); nbat += 1
        if (ep + 1) % max(1, epochs // 20) == 0 or ep == epochs - 1:
            vt = torch.from_numpy(vi)
            vl = _val(ema if diff is not None else model, diff, C[vt].to(dev), Y[vt].to(dev), is_ref)
            log(f'[{kind}] ep {ep + 1}/{epochs}  train {tl / max(nbat, 1):.4f}  val {vl:.4f}  lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    final = ema if diff is not None else model
    torch.save({'model': final.state_dict(), 'kind': kind}, out)
    log(f'[{kind}] saved {out}')
    return out


def _augment(c, y, kind):
    """Left-right flips. Valid for image-domain models; refiner/direct inputs are
    per-pair delay maps whose geometry is not mirror-symmetric pair-wise, so they are not flipped."""
    if kind in ('rd', 'dolce', 'unetR'):
        f = torch.rand(c.shape[0], device=c.device) < 0.5
        c = torch.where(f[:, None, None, None], c.flip(-1), c); y = torch.where(f[:, None, None, None], y.flip(-1), y)
    return c, y


@torch.no_grad()
def _val(model, diff, c, y, is_ref):
    model.eval()
    out = 0.0
    for s in range(0, len(y), 64):
        cc, yy = c[s:s + 64], y[s:s + 64]
        if diff is not None:
            torch.manual_seed(0)
            out += diff.loss(model, yy, cc, p_uncond=0.0).item() * len(yy)
        else:
            p = model(cc)
            out += (F.l1_loss(p, yy) if is_ref else F.mse_loss(p, yy)).item() * len(yy)
    return out / len(y)


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location='cpu')
    model, diff, _ = build(ck['kind'])
    model.load_state_dict(ck['model'])
    model = model.to(device).eval()
    if diff is not None: diff.to(device)
    return model, diff


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--out', required=True)
    ap.add_argument('--init', default=None); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=None); ap.add_argument('--lr_end', type=float, default=None)
    a = ap.parse_args()
    train(a.model, a.data, a.epochs, a.out, a.init, a.bs, a.lr, a.lr_end, log=lambda s: print(s, flush=True))

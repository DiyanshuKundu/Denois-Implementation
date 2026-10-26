"""Command-line entry points used by the Kaggle notebooks (one process per GPU).

    python -m sosbench.cli gen    --what S|C1|C1test|C2 --n N --seed S --out file.npz
    python -m sosbench.cli merge  --inputs a.npz b.npz --out c.npz
    python -m sosbench.cli real   --names abd3 abd4 phantom rat --root impact_data --out real.pkl
    python -m sosbench.cli eval   --ckpt ckpt --data data --real real.pkl --out results_denois
    python -m sosbench.cli thesis --method das|ssfm --name abd3 --root impact_data --out results_thesis
"""
import argparse, json, os, pickle, time
import numpy as np


def _backend():
    import torch
    from .backend import get_backend
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return get_backend('torch', dev) if dev == 'cuda' else get_backend('numpy')


def cmd_gen(a):
    from .acq import default_geometry
    from .datagen import Ops, gen_S, gen_C
    B = _backend(); geo = default_geometry()
    ops = Ops(B, geo, with_hi=(a.what == 'S'))
    t0 = time.time()
    if a.what == 'S':
        gen_S(ops, a.n, a.out, seed=a.seed, chunk=a.chunk)
    else:
        gen_C(B, ops, a.n, a.out, seed=a.seed, kind='structured' if a.what == 'C2' else 'quasi')
    print(f'done {a.what} n={a.n} in {(time.time() - t0) / 60:.1f} min', flush=True)


def cmd_merge(a):
    ds = [np.load(p) for p in a.inputs]
    np.savez(a.out, **{k: np.concatenate([d[k] for d in ds]) for k in ds[0].files})
    print('merged ->', a.out, sum(len(d['x']) for d in ds), 'samples')


def cmd_real(a):
    from .acq import default_geometry
    from .datagen import Ops, real_case
    B = _backend(); geo = default_geometry()
    ops = Ops(B, geo, with_hi=False)
    out = {}
    for n in a.names:
        t0 = time.time()
        c = real_case(B, ops, n, a.root)
        c.pop('geo')
        out[n] = c
        print(f'real {n}: {time.time() - t0:.0f}s', flush=True)
    with open(a.out, 'wb') as f:
        pickle.dump(out, f)


def _stack(d, idx=None):
    keys = ['x', 'b', 'w', 'l1', 'l2']
    sl = slice(None) if idx is None else idx
    return {k: np.asarray(d[k][sl]).astype(np.float32) for k in keys}


def cmd_eval(a):
    from .acq import default_geometry
    from .denois import DenoisRunner, score_grid, print_table
    from .metrics import all_metrics, roi_mask
    os.makedirs(a.out, exist_ok=True)
    geo = default_geometry()
    R = DenoisRunner(a.ckpt, geo, device='cuda')
    log = lambda s: print(s, flush=True)
    # --- tune the data-consistency weight nu on a few C1 validation cases (never on test data)
    pv = os.path.join(a.data, 'C1val.npz')
    val = _stack(np.load(pv)) if os.path.exists(pv) else _stack(np.load(os.path.join(a.data, 'C1.npz')), slice(-a.n_val, None))
    best = None
    for nu in a.nus:
        X = R.rd(val['b'], val['w'], val['l1'], val['l2'], nu=nu)
        r = np.mean([all_metrics(X[i], val['x'][i], geo)['RMSE'] for i in range(len(X))])
        log(f'nu={nu}: val RMSE {r:.2f}')
        if best is None or r < best[1]: best = (nu, r)
    R.nu = best[0]; log(f'selected nu={R.nu}')
    results = {'nu': R.nu}
    # --- simulated test sets
    for name in ['C1test', 'C2']:
        p = os.path.join(a.data, f'{name}.npz')
        if not os.path.exists(p): continue
        cs = _stack(np.load(p))
        log(f'== {name} ({len(cs["x"])} cases)')
        grid, obs = R.run_grid(cs, real=False, log=log)
        tab = score_grid(grid, cs['x'], geo)
        log(print_table(tab, 'RMSE', 'ACE'))
        results[name] = dict(table={f'{k[0]}|{k[1]}': v for k, v in tab.items()})
        np.savez_compressed(os.path.join(a.out, f'{name}_recons.npz'), gt=cs['x'],
                            **{f'{i}|{r}': X for i, rs in grid.items() for r, X in rs.items()})
    # --- real data (IMPACT): abd3/abd4 have GT, phantom / rat qualitative
    with open(a.real, 'rb') as f:
        real = pickle.load(f)
    for name, c in real.items():
        cs = {k: c[k][None].astype(np.float32) for k in ['x', 'b', 'w', 'l1', 'l2']}
        log(f'== {name}')
        grid, obs = R.run_grid(cs, real=(name in ('phantom', 'rat')), log=log)
        rec = {f'{i}|{r}': X[0] for i, rs in grid.items() for r, X in rs.items()}
        entry = {}
        if np.isfinite(c['x']).all():
            entry['metrics'] = {k: all_metrics(v, c['x'], geo) for k, v in rec.items()}
            for k, v in entry['metrics'].items():
                log(f'   {k:<28} RMSE {v["RMSE"]:6.2f}  MAE {v["MAE"]:6.2f}')
        results[name] = entry
        np.savez_compressed(os.path.join(a.out, f'{name}_recons.npz'), gt=c['x'], bmode=c['bmode'],
                            b=c['b'], w=c['w'], **rec, **{f'obs|{k}': v[0] for k, v in obs.items()})
    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    log('saved ' + a.out)


def cmd_thesis(a):
    import torch
    from .acq import load_impact, default_geometry
    from .metrics import all_metrics
    from .raymodel import resample_gt
    os.makedirs(a.out, exist_ok=True)
    data = load_impact(a.name, a.root)
    geo = default_geometry(elx=data['elx'])
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    if a.method == 'das':
        from .thesis_das import DASSetup, run_das_cmpe, ctrl_to_grid
        S = DASSetup(data)
        r = run_das_cmpe(S, device=a.devices[0], iters=a.iters, lam=a.lam_das, log=log)
        sos64 = ctrl_to_grid(r['c_ctrl'], r['ctrl_z'], r['ctrl_x'], geo.z, geo.x)
    else:
        from .thesis_ssfm import SSFMSetup, run_ssfm
        dov = min(40e-3, 0.95 * (data['t'][-1] * 1400 / 2))
        S = SSFMSetup(data, f_stride=a.f_stride, tx_stride=a.tx_stride, dov=dov)
        r = run_ssfm(S, devices=a.devices, iters_depth=a.iters // 3, iters_joint=a.iters - a.iters // 3, log=log)
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((r['z'], r['x']), r['sos'], bounds_error=False, fill_value=None)
        Z, X = np.meshgrid(np.clip(geo.z, r['z'][0], r['z'][-1]), geo.x, indexing='ij')
        sos64 = f(np.stack([Z.ravel(), X.ravel()], -1)).reshape(Z.shape).astype(np.float32)
    total = time.time() - t0
    out = dict(method=a.method, name=a.name, sos64=sos64, time_total_s=total,
               time_per_iter_s=r['time_per_iter'], peak_mem_gb=r['peak_mem_gb'], history=r['history'])
    if data['has_gt']:
        gt = resample_gt(data['C'], data['xC'], data['zC'], geo)
        out['gt'] = gt
        out['metrics'] = all_metrics(sos64, gt, geo)
        log(f"{a.method} {a.name}: RMSE {out['metrics']['RMSE']:.2f}  MAE {out['metrics']['MAE']:.2f}  "
            f"({total / 60:.1f} min, {r['peak_mem_gb']:.1f} GB)")
    with open(os.path.join(a.out, f'{a.method}_{a.name}.pkl'), 'wb') as f:
        pickle.dump(out, f)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen'); g.add_argument('--what', required=True); g.add_argument('--n', type=int, required=True)
    g.add_argument('--seed', type=int, default=0); g.add_argument('--out', required=True); g.add_argument('--chunk', type=int, default=256)
    m = sub.add_parser('merge'); m.add_argument('--inputs', nargs='+'); m.add_argument('--out')
    r = sub.add_parser('real'); r.add_argument('--names', nargs='+'); r.add_argument('--root', default='impact_data'); r.add_argument('--out')
    e = sub.add_parser('eval'); e.add_argument('--ckpt'); e.add_argument('--data'); e.add_argument('--real'); e.add_argument('--out')
    e.add_argument('--n_val', type=int, default=16); e.add_argument('--nus', type=float, nargs='+', default=[0.003, 0.01, 0.03, 0.1])
    t = sub.add_parser('thesis'); t.add_argument('--method', required=True); t.add_argument('--name', required=True)
    t.add_argument('--root', default='impact_data'); t.add_argument('--out', default='results_thesis')
    t.add_argument('--devices', nargs='+', default=['cuda:0']); t.add_argument('--iters', type=int, default=150)
    t.add_argument('--f_stride', type=int, default=8); t.add_argument('--tx_stride', type=int, default=1)
    t.add_argument('--lam_das', type=float, default=1e-3)
    a = ap.parse_args()
    dict(gen=cmd_gen, merge=cmd_merge, real=cmd_real, eval=cmd_eval, thesis=cmd_thesis)[a.cmd](a)


if __name__ == '__main__':
    main()

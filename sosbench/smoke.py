"""Fast end-to-end smoke test of every GPU code path (~3-6 min on a T4).
Run this first on Kaggle: it fails loudly if anything in the pipeline is broken."""
import time
import numpy as np


def run_smoke(root='impact_data', device='cuda'):
    import torch
    from .backend import get_backend
    from .acq import default_geometry, load_impact, observations_from_fsa
    from .datagen import Ops, simulate_case, set_fast_tv, TV_CFG
    from .wavesim import SimConfig
    from .phantoms import structured_sos
    from .nets import UNet, Diffusion, sos_norm
    from .solvers import dc_cg
    from .metrics import all_metrics
    from .raymodel import resample_gt
    import copy
    T = time.time()
    def ok(msg): print(f'[{time.time() - T:6.1f}s] OK  {msg}', flush=True)
    B = get_backend('torch', device); geo = default_geometry()
    saved = copy.deepcopy(TV_CFG); set_fast_tv(20)
    try:
        ops = Ops(B, geo, with_hi=True); ok('operators on GPU')
        rng = np.random.default_rng(0)
        hi = structured_sos(rng, 128, 128, geo.dz / 2, geo.dx / 2)
        cfg = SimConfig(geo)
        t0 = time.time(); d, r, w, comp = simulate_case(B, ops, cfg, rng, hi)
        assert np.isfinite(d).all() and d.shape == (8, 64, 64)
        ok(f'wave simulation + beamforming + tracking ({time.time() - t0:.1f}s/case, median rho {np.median(r):.2f})')
        l1 = ops.tv((d * w)[None], w[None], 'l1'); l2 = ops.tv((d * w)[None], w[None], 'l2')
        assert np.isfinite(l1).all() and np.isfinite(l2).all(); ok('L1-TV / L2-TV on GPU')
        # diffusion: one train step + DDIM with data consistency
        net = UNet(4, 1).to(device); diff = Diffusion(200, device)
        x0 = torch.from_numpy(sos_norm(hi[::2, ::2])[None, None].astype(np.float32)).to(device)
        cond = torch.cat([x0, x0], 1)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = diff.loss(net, x0.repeat(4, 1, 1, 1), cond.repeat(4, 1, 1, 1))
        loss.backward(); opt.step(); ok(f'diffusion train step (loss {loss.item():.3f})')
        from .denois import DenoisRunner
        R = DenoisRunner.__new__(DenoisRunner)
        R.dev, R.B, R.geo, R.ops, R.nu, R.cg_iters = device, B, geo, ops, 0.03, 5
        R.models = {'rd': (net.eval(), diff)}
        X = R.rd((d * w)[None], w[None], l1, l2)
        assert np.isfinite(X).all(); ok('R_D inference (DDIM + CG data consistency)')
        Xd = diff.ddpm(net, cond, (1, 1, 64, 64), dc=R._dc_fn((d * w)[None], w[None]), K=1)
        assert torch.isfinite(Xd).all(); ok('DDPM sampling with DC every step (DOLCE path)')
        # refiner U-Net forward/backward
        from .train import build, refiner_inputs
        mu, _, _ = build('mu_bPA'); mu = mu.to(device)
        Xin = torch.from_numpy(refiner_inputs((d * w)[None], w[None], l1, 'bPA')).to(device)
        out = mu(Xin); out.mean().backward(); ok(f'refiner M_U forward/backward {tuple(out.shape)}')
        # real data + thesis methods (1 iteration each)
        data = load_impact('abd3', root)
        dr, rr, bm = observations_from_fsa(data, default_geometry(elx=data['elx']), B); ok('real-data observations (abd3)')
        from .thesis_das import DASSetup, run_das_cmpe
        S = DASSetup(data); rd_ = run_das_cmpe(S, device=device, iters=2, log=lambda s: None)
        assert np.isfinite(rd_['c_ctrl']).all(); ok(f"DAS+CMPE 2 iters ({rd_['time_per_iter']:.1f}s/iter, {rd_['peak_mem_gb']:.1f} GB)")
        from .thesis_ssfm import SSFMSetup, run_ssfm
        S2 = SSFMSetup(data, f_stride=16, tx_stride=4)
        rs = run_ssfm(S2, devices=(device,), iters_depth=1, iters_joint=1, log=lambda s: None)
        assert np.isfinite(rs['sos']).all(); ok(f"SSFM 2 iters ({rs['time_per_iter']:.1f}s/iter, {rs['peak_mem_gb']:.1f} GB)")
    finally:
        TV_CFG.clear(); TV_CFG.update(saved)
    print('SMOKE TEST PASSED')

"""Builds the two self-contained Kaggle notebooks (package source embedded)."""
import json, os, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
PKG = {os.path.relpath(p, HERE): open(p).read() for p in sorted(glob.glob(os.path.join(HERE, 'sosbench', '*.py')))}


def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}
def code(s): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


SETUP = code('''
# Writes the `sosbench` package (all the code) into the working directory and checks the GPUs.
import os, sys, json, pickle, glob
WORK = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
os.chdir(WORK)
FILES = ''' + json.dumps(PKG) + '''
for fn, src in FILES.items():
    os.makedirs(os.path.dirname(os.path.join(WORK, fn)), exist_ok=True)
    with open(os.path.join(WORK, fn), 'w') as f:
        f.write(src)
sys.path.insert(0, WORK)
import numpy as np, torch, matplotlib.pyplot as plt
NGPU = torch.cuda.device_count()
print('torch', torch.__version__, '| GPUs:', [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, 'Enable a GPU accelerator (Settings -> Accelerator -> GPU T4 x2)'
''')

DOWNLOAD = code('''
# Public dataset: IMPACT release (Ali et al., IEEE TCI 2023) - ~1.8 GB, needs Internet ON.
#   abd3 / abd4 : k-Wave abdominal simulations WITH ground-truth SoS (the thesis' "Abdominal 3/4")
#   phantom     : real phantom (L12-5) with high-SoS alcohol inclusions (no pixel-wise GT)
#   rat         : in-vivo rat abdomen (L12-3v)                          (no GT)
from sosbench.acq import download_impact
IMPACT = 'impact_data'
download_impact(('abd3', 'abd4', 'phantom', 'rat'), IMPACT)
!ls -lh {IMPACT}
''')

SMOKE = code('''
# ~5 minute check that every GPU code path works (simulation, solvers, diffusion, refiner,
# DAS+CMPE and SSFM).  If this cell fails, nothing below will work - send me the traceback.
from sosbench.smoke import run_smoke
run_smoke(IMPACT)
''')

# ============================================================================ notebook 1
nb1 = [md('''
# DenOiS re-implementation on public ultrasound data (Kaggle, 2× T4)

**Paper:** Bezek & Goksel, *DenOiS: Dual-Domain Denoising of Observation and Solution in Ultrasound Image Reconstruction* (arXiv 2604.02105).
DenOiS reconstructs a **speed-of-sound (SoS) map** from pulse-echo ultrasound in two stages:
1. **Measurement refiner** `M` (U-Net `M_U` or diffusion `M_D`). It cleans and in-paints the noisy inter-transmit delay maps `b'`, using prior masks `P` and the projected first reconstruction `A'x'`.
2. **Reconstruction** `R_D`: a diffusion model conditioned on L1-TV / L2-TV solutions. It starts from a noised proposal and checks the data with conjugate gradients every K steps (plug-and-play).

**What this notebook does**

| Step | What | ≈ time (2×T4, `MODE='full'`) |
|---|---|---|
| 0 | write code, download data, smoke test | 10 min |
| 1 | sanity check: real k-Wave delays vs. the straight-ray model, and our wave simulator vs. k-Wave | 3 min |
| 2 | generate dataset **S** (10 000 ray-model cases) and **C** (620 + test sets, wave-simulated) | 30–40 min |
| 3 | train R_D, DOLCE, Unet(M_U), direct Unet, M_U ×3, M_D | 60–80 min |
| 4 | evaluate everything on simulated test sets + public data (Abdominal 3/4 with GT, phantom, rat) | 15–25 min |

Settings: **Accelerator → GPU T4 ×2**, **Internet → On**, then *Run All*. Use `MODE='quick'` first (≈25 min) to check the full pipeline end-to-end.

**Differences from the paper** (it has no public code or data):
* Public data instead of the authors' acquisitions: the **IMPACT** full-synthetic-aperture datasets. From these we synthesise the 9 transmits.
* Acquisition: 9 steered plane waves in [−16°, 16°] (8 MHz, 128 elements, 0.2 mm pitch), giving 8 delay maps on a 64×64 grid (0.6 × 0.4 mm). The paper used a 5 MHz, 0.3 mm-pitch probe.
* Dataset **C** is produced by our split-step Fourier (angular-spectrum) Born simulator instead of k-Wave. Step 1 checks it against k-Wave on Abdominal 3.
* Epoch counts are scaled to fit a Kaggle session. TV weights were tuned once on dataset S; the CG weight ν is tuned on a separate validation set. No test case is used for tuning.
'''),
SETUP,
code('''
MODE = 'full'            # 'quick' = tiny end-to-end run (~25 min) | 'full' = paper-style run
C = dict(
    full=dict(N_S=10000, N_C1=620, N_C1val=16, N_C1test=30, N_C2=30,
              ep=dict(rd_S=150, rd_C=100, dolce_S=150, dolce_C=100, unetR_S=100, unetR_C=100,
                      direct_S=100, direct_C=100, mu=150, md=150)),
    quick=dict(N_S=512, N_C1=40, N_C1val=4, N_C1test=6, N_C2=6,
               ep=dict(rd_S=3, rd_C=2, dolce_S=3, dolce_C=2, unetR_S=3, unetR_C=2,
                       direct_S=3, direct_C=2, mu=3, md=3)))[MODE]
DATA, CKPT, RES = 'data', 'ckpt', 'results_denois'
for d in (DATA, CKPT, RES): os.makedirs(d, exist_ok=True)
GPU1 = 1 if NGPU > 1 else 0
PY = sys.executable
'''),
DOWNLOAD,
SMOKE,
md('''
## Step 1 — Sanity check on real data

Before training anything, check the two physical building blocks on **Abdominal 3**, which has ground truth:
* **Measured vs. ray model.** We measure delay maps from the k-Wave channel data and compare them with what the straight-ray operator `A'` predicts from the true SoS map. DenOiS assumes they are related but imperfect. Expect a positive correlation (~0.4–0.8), with the model over-predicting.
* **Our simulator vs. k-Wave.** We re-simulate the same medium with our wave simulator. Its delays should resemble k-Wave's (correlation ~0.6–0.75). That justifies using it for dataset **C**.
'''),
code('''
from sosbench.backend import get_backend
from sosbench.acq import default_geometry, load_impact, observations_from_fsa
from sosbench.raymodel import build_operator, aperture_mask, sos_to_u, resample_gt
from sosbench.datagen import Ops, simulate_case
from sosbench.wavesim import SimConfig
from sosbench.viz import show_obs, show_maps
B = get_backend('torch', 'cuda:0')
d3 = load_impact('abd3', IMPACT)
geo3 = default_geometry(elx=d3['elx'])
dl, rho, bm = observations_from_fsa(d3, geo3, B)
A = build_operator(geo3); gt3 = resample_gt(d3['C'], d3['xC'], d3['zC'], geo3)
bp = (A @ sos_to_u(gt3).ravel()).reshape(8, 64, 64); m = aperture_mask(geo3)
# our simulator on the same medium (upsampled GT on the 128x128 grid)
ops = Ops(B, default_geometry(), with_hi=False)
gt3_hi = resample_gt(d3['C'], d3['xC'], d3['zC'], geo3, upsample=2)
ds, rs, ws, _ = simulate_case(B, ops, SimConfig(ops.geo), np.random.default_rng(0), gt3_hi)
print('pair | corr(k-Wave, ray model) | slope | corr(our sim, k-Wave)')
for k in range(8):
    mm = m[k] & (rho[k] > 0.5) & (rs[k] > 0.5)
    print(f'{k:4d} | {np.corrcoef(dl[k][mm], bp[k][mm])[0,1]:24.2f} | {np.polyfit(bp[k][mm], dl[k][mm], 1)[0]:5.2f} | {np.corrcoef(ds[k][mm], dl[k][mm])[0,1]:20.2f}')
show_obs({'k-Wave measured': dl, "ray model A'x (GT)": bp, 'our simulator': ds}, geo3, w=m.astype(float), title='Abdominal 3: inter-transmit delays [ns]'); plt.show()
show_maps({}, geo3, gt=gt3, bmode=bm); plt.show()
'''),
md('''
**Why the abdominal layers are hard for DenOiS-style data (worth knowing before reading the results).**
The cell below asks how much of the delay signal comes from the purely *depth-layered* part of the abdomen. It also runs an "oracle" L2-TV on perfect, noise-free model data. If the layered fraction is small, delay-based acquisitions are nearly blind to horizontal layers. Then no method that uses only these delays can recover them, except by guessing from its training prior.
'''),
code('''
from sosbench.solvers import LinearOp, tv_irls, U_SCALE
from sosbench.raymodel import u_to_sos
from sosbench.metrics import all_metrics
op = LinearOp(B, A, 64, 64)
lay = np.tile(gt3.mean(1, keepdims=True), (1, 64))
wm = B.asarray(m.reshape(1, -1).astype(np.float32))
e_full = float((wm * op.fwd(B.asarray(sos_to_u(gt3)[None] / U_SCALE))).norm())
e_lay = float((wm * op.fwd(B.asarray(sos_to_u(lay)[None] / U_SCALE))).norm())
print(f'fraction of delay-signal energy explained by depth layering: {e_lay / e_full:.3f}')
bn = wm * op.fwd(B.asarray(sos_to_u(gt3)[None] / U_SCALE))
u = tv_irls(op, bn, wm, lam=3e-5, p=2, outer=8, inner=150)
oracle = u_to_sos(B.to_numpy(u)[0] * U_SCALE)
print('oracle L2-TV from PERFECT data:', {k: round(v, 1) for k, v in all_metrics(oracle, gt3, geo3).items()})
print('constant 1540 m/s            :', {k: round(v, 1) for k, v in all_metrics(np.full_like(gt3, 1540), gt3, geo3).items()})
'''),
md('''
## Step 2 — Datasets S and C (both GPUs in parallel)
* **S**: 10 000 quasi-random SoS maps in [1400, 1650] m/s. Some are homogeneous, 50% are blurred, and 30% have tissue-like layers. Delays come from a 2× finer operator `A_H` (discretisation mismatch), plus noise, holes and outliers.
* **C1** (620 training cases) and **C1test** / **C1val**: wave-simulated, with the full DAS + tracking pipeline. This means realistic noise, missing data and model mismatch.
* **C2** (30): well-structured circular / rectangular inclusions of positive and negative contrast. This is a test set with a different distribution, as in the paper.
'''),
code('''
from sosbench.jobs import start_queue, wait_all
g = f'{PY} -m sosbench.cli gen'
h = lambda n: (n // 2, n - n // 2)
s0, s1 = h(C['N_S']); c0, c1 = h(C['N_C1'])
q0 = [f'{g} --what S --n {s0} --seed 0 --out {DATA}/S_0.npz',
      f'{g} --what C1 --n {c0} --seed 10 --out {DATA}/C1_0.npz',
      f'{g} --what C1test --n {C["N_C1test"]} --seed 99 --out {DATA}/C1test.npz']
q1 = [f'{g} --what S --n {s1} --seed 1 --out {DATA}/S_1.npz',
      f'{g} --what C1 --n {c1} --seed 11 --out {DATA}/C1_1.npz',
      f'{g} --what C2 --n {C["N_C2"]} --seed 98 --out {DATA}/C2.npz',
      f'{g} --what C1val --n {C["N_C1val"]} --seed 97 --out {DATA}/C1val.npz']
wait_all([start_queue(q0, 0, 'log_gen0.txt', WORK), start_queue(q1, GPU1, 'log_gen1.txt', WORK)])
!{PY} -m sosbench.cli merge --inputs {DATA}/S_0.npz {DATA}/S_1.npz --out {DATA}/S.npz
!{PY} -m sosbench.cli merge --inputs {DATA}/C1_0.npz {DATA}/C1_1.npz --out {DATA}/C1.npz
'''),
code('''
# A few examples: ground truth, L1-TV, L2-TV from the observed delays
from sosbench.metrics import all_metrics
geo = default_geometry()
for nm in ['S', 'C1', 'C2']:
    dd = np.load(f'{DATA}/{nm}.npz')
    for i in range(2):
        show_maps({'L1-TV': dd['l1'][i], 'L2-TV': dd['l2'][i]}, geo, gt=dd['x'][i], title=f'{nm} #{i}', size=2.2); plt.show()
'''),
md('''
## Step 3 — Training (two queues, one per GPU)
GPU 0: `R_D` (pre-train on S, fine-tune on C1), then the three U-Net refiners `M_U(b')`, `M_U(b',P)` and `M_U(b',P,A'x')`.
GPU 1: DOLCE, Unet(M_U), the direct Unet baseline, and the diffusion refiner `M_D(b',P,A'x')`.
'''),
code('''
t = f'{PY} -m sosbench.train'
E = C['ep']
q0 = [f'{t} --model rd --data {DATA}/S.npz --epochs {E["rd_S"]} --out {CKPT}/rd_S.pt',
      f'{t} --model rd --data {DATA}/C1.npz --epochs {E["rd_C"]} --init {CKPT}/rd_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/rd.pt',
      f'{t} --model mu_b --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_b.pt',
      f'{t} --model mu_bP --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_bP.pt',
      f'{t} --model mu_bPA --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_bPA.pt']
q1 = [f'{t} --model dolce --data {DATA}/S.npz --epochs {E["dolce_S"]} --out {CKPT}/dolce_S.pt',
      f'{t} --model dolce --data {DATA}/C1.npz --epochs {E["dolce_C"]} --init {CKPT}/dolce_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/dolce.pt',
      f'{t} --model unetR --data {DATA}/S.npz --epochs {E["unetR_S"]} --out {CKPT}/unetR_S.pt',
      f'{t} --model unetR --data {DATA}/C1.npz --epochs {E["unetR_C"]} --init {CKPT}/unetR_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/unetR.pt',
      f'{t} --model direct --data {DATA}/S.npz --epochs {E["direct_S"]} --out {CKPT}/direct_S.pt',
      f'{t} --model direct --data {DATA}/C1.npz --epochs {E["direct_C"]} --init {CKPT}/direct_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/direct.pt',
      f'{t} --model md_bPA --data {DATA}/C1.npz --epochs {E["md"]} --out {CKPT}/md_bPA.pt']
wait_all([start_queue(q0, 0, 'log_train0.txt', WORK), start_queue(q1, GPU1, 'log_train1.txt', WORK)], every=120)
'''),
md('''
## Step 4 — Evaluation
The same grid as the paper's Tables 1–2: **rows** = what the reconstruction is fed (raw `b'` or refined `M(·)`), **columns** = reconstruction method.
Metrics: RMSE [m/s] over the field of view, and in brackets ACE, the absolute contrast error of the inclusion against a 5 mm ring around it. ACE is the clinically relevant one. We also report gCNR (detectability).
'''),
code('''
!{PY} -m sosbench.cli real --names abd3 abd4 phantom rat --root {IMPACT} --out {DATA}/real.pkl
!{PY} -m sosbench.cli eval --ckpt {CKPT} --data {DATA} --real {DATA}/real.pkl --out {RES}
'''),
code('''
from sosbench.viz import table_frame
from IPython.display import display, Markdown
res = json.load(open(f'{RES}/results.json'))
print('selected data-consistency weight nu =', res['nu'])
TABS = {}
for name, title in [('C2', 'C2 - structured inclusions (paper Table 1 analogue)'), ('C1test', 'C1test - quasi-random, held out')]:
    if name not in res: continue
    tab = {tuple(k.split('|')): {m: tuple(v) for m, v in d.items()} for k, d in res[name]['table'].items()}
    TABS[name] = tab
    display(Markdown(f'### {title}'))
    display(table_frame(tab, ('RMSE', 'ACE', 'gCNR')))
display(Markdown("""**Paper reference (their data, Table 1, C2, RMSE (ACE))**: L2-TV 11.95 (36.25), L1-TV 10.39 (32.37), Dolce 6.75 (15.73), R_D 6.78 (18.31); with refinement M_U(b',P,A'x') + R_D 7.55 (11.53).
Compare the *trends*, not the absolute numbers: the probe, data and simulator differ."""))
'''),
code('''
# Abdominal 3 / 4 (k-Wave, ground truth): every input x reconstruction combination
for name in ['abd3', 'abd4']:
    if 'metrics' not in res.get(name, {}): continue
    rows = {k: {'RMSE': v['RMSE'], 'MAE': v['MAE']} for k, v in res[name]['metrics'].items()}
    import pandas as pd
    df = pd.DataFrame(rows).T.sort_values('RMSE')
    display(Markdown(f'### {name}: SoS error vs ground truth (thesis ROI x in [-12,12] mm, z in [0,38.4] mm)'))
    display(df.round(2))
'''),
code('''
# Figures: sample reconstructions (C2), the public cases, and refined observations
SHOW = ["b'|L2-TV", "b'|L1-TV", "b'|Unet[5]", "b'|Dolce", "MU(b',P,A'x')|Unet(MU)", "b'|RD",
        "MU(b',P,A'x')|RD", "MD(b',P,A'x')|RD"]
def pick(z):
    return {k.replace("|", "\\n"): z[k] for k in SHOW if k in z.files}
geo = default_geometry()
if os.path.exists(f'{RES}/C2_recons.npz'):
    z = np.load(f'{RES}/C2_recons.npz')
    for i in range(min(3, len(z['gt']))):
        show_maps({k: v[i] for k, v in pick(z).items()}, geo, gt=z['gt'][i], ncol=9, size=2.0, title=f'C2 case {i}'); plt.show()
for name in ['abd3', 'abd4', 'phantom', 'rat']:
    p = f'{RES}/{name}_recons.npz'
    if not os.path.exists(p): continue
    z = np.load(p)
    show_maps(pick(z), geo, gt=z['gt'], ncol=9, size=2.0, title=name, vmin=1420, vmax=1640); plt.show()
    obs = {"acquired b'": z['b']}
    obs.update({k[4:]: z[k] for k in z.files if k.startswith('obs|')})
    show_obs(obs, geo, w=z['w'], title=f'{name}: acquired vs refined delays'); plt.show()
'''),
code('''
# "Is DenOiS working?" - automatic checks of the paper's main claims on OUR runs
def get(tab, i, r, m):
    v = tab.get((i, r)); return None if v is None or m not in v else v[m][0]
checks = []
for name, tab in TABS.items():
    ref = "MU(b',P,A'x')"
    for metric in ('RMSE', 'ACE'):
        a, b_ = get(tab, "b'", 'L1-TV', metric), get(tab, ref, 'L1-TV', metric)
        if a and b_: checks.append((name, f'refinement improves L1-TV {metric}', b_ < a, f'{a:.2f} -> {b_:.2f}'))
    for base in ('L1-TV', 'L2-TV', 'Dolce', 'Unet(MU)'):
        best_rd = min([x for x in [get(tab, ref, 'RD', 'ACE'), get(tab, "MD(b',P,A'x')", 'RD', 'ACE')] if x is not None], default=None)
        bb = min([x for x in [get(tab, i_, base, 'ACE') for i_ in ["b'", ref]] if x is not None], default=None)
        if best_rd and bb: checks.append((name, f'DenOiS R_D beats {base} on ACE', best_rd < bb, f'{bb:.2f} vs {best_rd:.2f}'))
    u5 = get(tab, "b'", 'Unet[5]', 'ACE'); rd0 = get(tab, "b'", 'RD', 'ACE')
    if u5 and rd0: checks.append((name, 'direct Unet[5] loses contrast (paper: it fails)', u5 > rd0, f'ACE {u5:.2f} vs R_D {rd0:.2f}'))
for name in ['abd3', 'abd4']:
    mt = res.get(name, {}).get('metrics', {})
    a, b_ = mt.get("b'|L1-TV"), mt.get("MU(b',P,A'x')|RD")
    if a and b_: checks.append((name, 'DenOiS R_D(M_U) beats L1-TV (RMSE)', b_['RMSE'] < a['RMSE'], f"{a['RMSE']:.2f} -> {b_['RMSE']:.2f}"))
import pandas as pd
display(pd.DataFrame(checks, columns=['data', 'claim', 'holds', 'numbers']))
json.dump({'checks': [(a, b, bool(c), d) for a, b, c, d in checks]}, open(f'{RES}/checks.json', 'w'), indent=1)
print('Results saved in', os.path.abspath(RES), '- notebook 2 reads them from here (or from this notebook\\'s output).')
'''),
md('''
### How to read these results
* **Simulated test sets (C2, C1test)** are DenOiS's home ground (inclusions, the same probe as training). The paper's claims should show up here: refinement lowers RMSE/ACE for the analytical methods, R_D beats DOLCE and Unet(M_U) on ACE, and the direct Unet loses contrast.
* **Abdominal 3/4** is a stress test. Step 1 showed these layered media sit mostly in the blind spot of delay-based acquisitions. Expect every method here to stay near the "constant 1540" error (~34 / ~30 m/s), unless the learned prior guesses the layers.
* **Phantom / rat** have no pixel-wise ground truth. Judge them visually: do the phantom's high-SoS inclusions appear where the B-mode shows them, and are there fewer false blobs than with L1/L2-TV?
'''),
]

# ============================================================================ notebook 2
nb2 = [md('''
# Thesis methods (DAS+CMPE, SSFM) on the same public data — and comparison with DenOiS

**Thesis:** H. Meng, *Differentiable Wave Propagation for Ultrasound Imaging of Speed-of-Sound* (MSc, Uppsala 2026).
* **DAS+CMPE**: straight-ray delay-and-sum. It is optimised so that common-midpoint sub-aperture images agree in phase (Simson et al.).
* **SSFM**: Fourier split-step wave propagation. It is optimised to maximise the focusing quality of the reflection matrix (Ali et al. / Lambert et al.).

Both are **unsupervised**: they need no training data and are optimised per case from 1540 m/s.
The thesis used exactly these public **Abdominal 3/4** cases, so we can check against its Table 5 (Abd3: SSFM 18.60 vs DAS 34.40 m/s RMSE; Abd4: 26.14 vs 30.02).

**Run order:** run notebook 1 (DenOiS) first. Then either run this one in the same session, or add notebook 1's *output* as an input (Add Input → Notebooks), so the DenOiS results can be found.

T4 adaptations (the thesis used a 96 GB GPU): SSFM keeps every 8th frequency bin (the thesis's own stride study shows little loss), uses gradient checkpointing over depth, and splits the transmit elements across the two T4s. Everything else (150 iterations = 50 depth-only + 100 joint, TV αx=5 / αz=2, 3×3 partial median, Adam) follows the thesis.
'''),
SETUP,
code('''
IMPACT = 'impact_data'; RES2 = 'results_thesis'; os.makedirs(RES2, exist_ok=True)
PY = sys.executable
MODE = 'full'   # 'quick' -> 10 iterations per method (pipeline check)
ITERS = 150 if MODE == 'full' else 10
'''),
DOWNLOAD,
SMOKE,
md('''
## Run both methods on all four public cases
GPU 0 runs DAS+CMPE on all cases. GPU 1 runs SSFM. SSFM uses both GPUs (transmit split) when it runs on its own.
'''),
code('''
from sosbench.jobs import start_queue, wait_all
th = f'{PY} -m sosbench.cli thesis --root {IMPACT} --out {RES2} --iters {ITERS}'
cases = ['abd3', 'abd4', 'phantom', 'rat']
if NGPU > 1:
    q0 = [f'{th} --method das --name {n}' for n in cases]
    q1 = [f'{th} --method ssfm --name {n}' for n in cases]
    wait_all([start_queue(q0, 0, 'log_das.txt', WORK), start_queue(q1, 1, 'log_ssfm.txt', WORK)], every=120)
else:
    wait_all([start_queue([f'{th} --method {m} --name {n}' for m in ('das', 'ssfm') for n in cases], 0, 'log_thesis.txt', WORK)], every=120)
'''),
code('''
import pandas as pd
from IPython.display import display, Markdown
from sosbench.acq import default_geometry
from sosbench.viz import show_maps
from sosbench.metrics import all_metrics
TH = {}
for f in sorted(glob.glob(f'{RES2}/*.pkl')):
    r = pickle.load(open(f, 'rb')); TH[(r['method'], r['name'])] = r
THESIS_REPORTED = {('ssfm', 'abd3'): (18.60, 15.55), ('das', 'abd3'): (34.40, 25.65),
                   ('ssfm', 'abd4'): (26.14, 17.16), ('das', 'abd4'): (30.02, 25.26)}
rows = []
for (m, n), r in TH.items():
    row = dict(method={'das': 'DAS+CMPE', 'ssfm': 'SSFM'}[m], case=n, time_min=r['time_total_s'] / 60,
               s_per_iter=r['time_per_iter_s'], peak_GB=r['peak_mem_gb'])
    if 'metrics' in r:
        row.update(RMSE=r['metrics']['RMSE'], MAE=r['metrics']['MAE'])
        if (m, n) in THESIS_REPORTED:
            row.update(thesis_RMSE=THESIS_REPORTED[(m, n)][0], thesis_MAE=THESIS_REPORTED[(m, n)][1])
    rows.append(row)
display(Markdown('### Thesis methods: our reproduction vs. numbers reported in the thesis (Table 5)'))
display(pd.DataFrame(rows).round(2))
for (m, n), r in TH.items():
    h = pd.DataFrame(r['history'])
    plt.plot(h['loss'].values, label=f'{m} {n}')
plt.xlabel('iteration'); plt.ylabel('objective'); plt.legend(fontsize=7); plt.title('optimisation curves'); plt.show()
'''),
md('''
## Head-to-head: DenOiS vs. the thesis methods
All methods are evaluated on the **same 64×64 grid and the same ROI** (x ∈ [−12, 12] mm, z ∈ [0, 38.4] mm, as in the thesis).
'''),
code('''
cand = [os.path.join(WORK, 'results_denois')] + sorted(glob.glob('/kaggle/input/**/results_denois', recursive=True))
DEN = next((c for c in cand if os.path.exists(os.path.join(c, 'results.json'))), None)
print('DenOiS results:', DEN)
geo = default_geometry()
PICK = {'Constant 1540': None, 'L2-TV': "b'|L2-TV", 'L1-TV': "b'|L1-TV", 'DOLCE': "b'|Dolce",
        'DenOiS R_D (raw b\\')': "b'|RD", 'DenOiS R_D(M_U)': "MU(b',P,A'x')|RD", 'DenOiS R_D(M_D)': "MD(b',P,A'x')|RD"}
summary = []; MAPS = {}
for n in ['abd3', 'abd4', 'phantom', 'rat']:
    maps = {}
    gt = None
    if DEN and os.path.exists(f'{DEN}/{n}_recons.npz'):
        z = np.load(f'{DEN}/{n}_recons.npz'); gt = z['gt']
        for lab, k in PICK.items():
            if k is None: maps[lab] = np.full((64, 64), 1540.0, np.float32)
            elif k in z.files: maps[lab] = z[k]
    for m, lab in (('das', 'DAS+CMPE (thesis)'), ('ssfm', 'SSFM (thesis)')):
        if (m, n) in TH:
            maps[lab] = TH[(m, n)]['sos64']
            if gt is None and 'gt' in TH[(m, n)]: gt = TH[(m, n)]['gt']
    MAPS[n] = (maps, gt)
    if gt is not None and np.isfinite(gt).all():
        for lab, X in maps.items():
            mt = all_metrics(X, gt, geo); summary.append(dict(case=n, method=lab, RMSE=mt['RMSE'], MAE=mt['MAE']))
S = pd.DataFrame(summary)
if len(S):
    piv = S.pivot(index='method', columns='case', values=['RMSE', 'MAE'])
    piv[('RMSE', 'mean')] = piv['RMSE'].mean(1); piv = piv.sort_values(('RMSE', 'mean'))
    display(piv.round(2))
for n, (maps, gt) in MAPS.items():
    if maps:
        show_maps(maps, geo, gt=gt, ncol=9, size=2.0, title=n, vmin=1420, vmax=1640, metric_keys=('RMSE', 'MAE')); plt.show()
'''),
code('''
# Simulated inclusion test sets: only DenOiS can be scored there (the thesis methods need
# full-synthetic-aperture data, which our 9-transmit simulator does not produce).
if DEN:
    res = json.load(open(f'{DEN}/results.json'))
    for name in ('C2', 'C1test'):
        if name in res:
            t = {k: v for k, v in res[name]['table'].items()}
            best = sorted(((v['ACE'][0] if 'ACE' in v else 1e9, v['RMSE'][0], k) for k, v in t.items()))[:6]
            display(Markdown(f'**{name}: best 6 DenOiS-grid combinations by ACE** (ACE, RMSE): ' +
                             '; '.join(f'`{k}` {a:.1f}, {r:.1f}' for a, r, k in best)))
    if os.path.exists(f'{DEN}/checks.json'):
        display(pd.DataFrame(json.load(open(f'{DEN}/checks.json'))['checks'], columns=['data', 'claim', 'holds', 'numbers']))
'''),
md('''
## Conclusion (generated from the numbers above)
'''),
code('''
lines = []
if len(S):
    mean = S.groupby('method')['RMSE'].mean().sort_values()
    lines.append(f"* On the ground-truth abdominal cases the lowest mean RMSE is **{mean.index[0]}** ({mean.iloc[0]:.1f} m/s), "
                 f"vs. {mean.get('Constant 1540', float('nan')):.1f} m/s for a constant 1540 m/s guess.")
    for m in ('SSFM (thesis)', 'DAS+CMPE (thesis)'):
        for n, ref in (('abd3', {'SSFM (thesis)': 18.60, 'DAS+CMPE (thesis)': 34.40}), ('abd4', {'SSFM (thesis)': 26.14, 'DAS+CMPE (thesis)': 30.02})):
            v = S[(S.method == m) & (S.case == n)]['RMSE']
            if len(v): lines.append(f"* {m} on {n}: {v.iloc[0]:.1f} m/s here vs {ref[m]:.1f} m/s reported in the thesis.")
    dn = [m for m in mean.index if m.startswith('DenOiS')]
    if dn: lines.append(f"* Best DenOiS variant on the abdomen: {dn[0]} ({mean[dn[0]]:.1f} m/s).")
if DEN and os.path.exists(f'{DEN}/checks.json'):
    ch = json.load(open(f'{DEN}/checks.json'))['checks']
    lines.append(f"* DenOiS paper claims reproduced on simulated inclusion data: {sum(c[2] for c in ch)}/{len(ch)} checks hold.")
display(Markdown('\\n'.join(lines) if lines else 'Run the cells above first.'))
'''),
md('''
### Interpreting the comparison
* **The two approaches answer different questions.** DenOiS learns priors from simulations and works from 9 transmits (a fast, clinically practical acquisition). Its natural targets are *focal inclusions* (lesions), scored by contrast (ACE, gCNR). The thesis methods work per-case on full-synthetic-aperture data, need no training, and are scored by pixel-wise SoS error on layered abdominal media.
* **Layered media favour the wave-based thesis method.** The Step 1 analysis in notebook 1 shows that the depth-layered part of the abdomen produces only a few percent of the inter-transmit delay signal. That information simply isn't in DenOiS's input. SSFM uses the full aperture and a wave model, and can recover it.
* **Inclusions favour DenOiS.** On C2 and C1test, check whether refinement and the diffusion prior reduce contrast error (ACE) compared with TV, DOLCE and Unet(M_U), as the paper claims.
* **Combining them** (see the earlier discussion): use DenOiS's map as SSFM's starting point instead of 1540 m/s, or replace SSFM's hand-made TV + median filter with DenOiS's diffusion prior.
* **Limitations of this study:** a single public probe; DenOiS's training data is simulated with our split-step simulator, not k-Wave; few real cases; and the thesis methods were not run on the simulated inclusion sets.
'''),
]


def write(cells, path):
    import copy; cells = copy.deepcopy(cells)
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'},
                                       'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True, 'isGpuEnabled': True}},
          'nbformat': 4, 'nbformat_minor': 5}
    for c in nb['cells']:
        src = c['source']; c['source'] = [l + '\n' for l in src.split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
    json.dump(nb, open(path, 'w'), indent=1)
    print('wrote', path, os.path.getsize(path) // 1024, 'KB')


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'dist'), exist_ok=True)
    write(nb1, os.path.join(HERE, 'dist', '01_denois_reproduction.ipynb'))
    write(nb2, os.path.join(HERE, 'dist', '02_thesis_methods_vs_denois.ipynb'))

"""Regenerates notebooks/sos_denois_vs_thesis.ipynb from the sosbench sources.
Usage (from the repository root):  python notebooks/build_merged.py

Single merged Kaggle notebook: DenOiS reproduction + thesis methods + comparison,
with a reduced time budget (~2-2.5 h on 2x T4)."""
import os
import build_nb as B
from build_nb import md, code

nb1, nb2 = B.nb1, B.nb2

INTRO = md('''
# DenOiS vs. thesis methods: speed-of-sound imaging on public ultrasound data (one notebook, 2× T4)

**Part A: DenOiS** (Bezek & Goksel 2026). A refiner network `M` cleans up the measured delays, then the diffusion reconstruction `R_D` rebuilds the speed-of-sound (SoS) map with plug-and-play data consistency.
**Part B: thesis methods** (Meng 2026). DAS+CMPE (straight-ray) and SSFM (wave-based). Both are unsupervised and optimised separately for each case.
**Part C: head-to-head comparison** on the same public data (IMPACT: Abdominal 3/4 with ground truth, a real phantom, a rat in vivo), plus a conclusion generated from the numbers.

**Time budget: about 2–2.5 h** (`MODE='full'`). To get there:
* DenOiS trains on 3 000 ray-model cases (S) and 300 wave-simulated cases (C1), with 20–40 epochs. The paper used 10 000 / 620 cases and 400 epochs.
* The thesis methods **run in the background while DenOiS generates data and trains**, sharing the GPUs.
* SSFM keeps every 16th frequency bin and every 4th transmit element (≈5 s/iteration). It still runs 150 iterations (50 depth-only + 100 joint), as in the thesis.

**Settings:** Accelerator **GPU T4 ×2**, Internet **On**. Then *Save Version → Save & Run All (Commit)*. Use `MODE='quick'` (≈20 min) for a pipeline check only. Its numbers are meaningless.
Optional: fill in the GitHub settings in the last cell to push the results there automatically.
''')

CONFIG = code('''
MODE = 'full'      # 'full' (~2-2.5 h) | 'quick' (~20 min pipeline check, meaningless numbers)
C = dict(
    full=dict(N_S=3000, N_C1=300, N_C1val=16, N_C1test=30, N_C2=30, THESIS_ITERS=150,
              ep=dict(rd_S=40, rd_C=30, dolce_S=40, dolce_C=30, unetR_S=20, unetR_C=20,
                      direct_S=20, direct_C=20, mu=30, md=30)),
    quick=dict(N_S=512, N_C1=40, N_C1val=4, N_C1test=6, N_C2=6, THESIS_ITERS=10,
               ep=dict(rd_S=3, rd_C=2, dolce_S=3, dolce_C=2, unetR_S=3, unetR_C=2,
                       direct_S=3, direct_C=2, mu=3, md=3)))[MODE]
DATA, CKPT, RES, RES2 = 'data', 'ckpt', 'results_denois', 'results_thesis'
for d in (DATA, CKPT, RES, RES2): os.makedirs(d, exist_ok=True)
GPU1 = 1 if NGPU > 1 else 0
PY = sys.executable
import time; T_START = time.time()
''')

THESIS_LAUNCH = [md('''
## Part B starts now, in the background
DAS+CMPE and SSFM run on all four public cases **while** Part A generates data and trains. Logs: `log_thesis0.txt`, `log_thesis1.txt`. We collect their results in Part C.
'''), code('''
from sosbench.jobs import start_queue, wait_all
th = f'{PY} -m sosbench.cli thesis --root {IMPACT} --out {RES2} --iters {C["THESIS_ITERS"]}'
ss = '--f_stride 16 --tx_stride 4'          # T4-friendly SSFM sampling (~5 s / iteration)
thesis_q0 = [f'{th} --method ssfm --name abd3 {ss}', f'{th} --method das --name abd3', f'{th} --method das --name phantom',
             f'{th} --method ssfm --name rat {ss}']
thesis_q1 = [f'{th} --method ssfm --name abd4 {ss}', f'{th} --method das --name abd4', f'{th} --method das --name rat',
             f'{th} --method ssfm --name phantom {ss}']
THESIS_PROCS = [start_queue(thesis_q0, 0, 'log_thesis0.txt', WORK), start_queue(thesis_q1, GPU1, 'log_thesis1.txt', WORK)]
print('thesis queues started in the background')
''')]

DATAGEN_MD = md('''
## Part A, Step 2: datasets S and C (both GPUs)
* **S**: 3 000 quasi-random SoS maps in [1400, 1650] m/s, some with tissue-like layers. Delays come from a 2× finer ray operator, plus noise, holes and outliers.
* **C1** (300 training cases), **C1val** (16) and **C1test** (30) are wave-simulated with the full DAS + tracking pipeline. **C2** (30) has structured circular/rectangular inclusions, a test set with a different distribution from training.
''')

TRAIN_MD = md('''
## Part A, Step 3: training (two queues, one per GPU; the thesis jobs share the GPUs)
GPU 0: `R_D` (pre-train on S, then fine-tune on C1), the diffusion refiner `M_D`, and the three U-Net refiners `M_U`.
GPU 1: DOLCE, Unet(M_U) and the direct-Unet baseline.
''')

TRAIN = code('''
t = f'{PY} -m sosbench.train'
E = C['ep']
q0 = [f'{t} --model rd --data {DATA}/S.npz --epochs {E["rd_S"]} --out {CKPT}/rd_S.pt',
      f'{t} --model rd --data {DATA}/C1.npz --epochs {E["rd_C"]} --init {CKPT}/rd_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/rd.pt',
      f'{t} --model md_bPA --data {DATA}/C1.npz --epochs {E["md"]} --out {CKPT}/md_bPA.pt',
      f'{t} --model mu_b --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_b.pt',
      f'{t} --model mu_bP --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_bP.pt',
      f'{t} --model mu_bPA --data {DATA}/C1.npz --epochs {E["mu"]} --out {CKPT}/mu_bPA.pt']
q1 = [f'{t} --model dolce --data {DATA}/S.npz --epochs {E["dolce_S"]} --out {CKPT}/dolce_S.pt',
      f'{t} --model dolce --data {DATA}/C1.npz --epochs {E["dolce_C"]} --init {CKPT}/dolce_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/dolce.pt',
      f'{t} --model unetR --data {DATA}/S.npz --epochs {E["unetR_S"]} --out {CKPT}/unetR_S.pt',
      f'{t} --model unetR --data {DATA}/C1.npz --epochs {E["unetR_C"]} --init {CKPT}/unetR_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/unetR.pt',
      f'{t} --model direct --data {DATA}/S.npz --epochs {E["direct_S"]} --out {CKPT}/direct_S.pt',
      f'{t} --model direct --data {DATA}/C1.npz --epochs {E["direct_C"]} --init {CKPT}/direct_S.pt --lr 3e-4 --lr_end 1e-5 --out {CKPT}/direct.pt']
wait_all([start_queue(q0, 0, 'log_train0.txt', WORK), start_queue(q1, GPU1, 'log_train1.txt', WORK)], every=180)
print(f'total elapsed so far: {(time.time() - T_START) / 60:.0f} min')
''')

THESIS_WAIT = [md('''
# Part B: thesis methods results
Wait for the background DAS+CMPE / SSFM jobs (usually already finished), then check them against the thesis's Table 5.
'''), code('''
try:
    wait_all(THESIS_PROCS, every=60)
except RuntimeError as e:          # keep going: compare whatever finished, the log above shows what failed
    print('WARNING:', e)
print(f'total elapsed so far: {(time.time() - T_START) / 60:.0f} min')
''')]

GITHUB = [md('''
## Save results to GitHub (optional)
Add a Kaggle secret `GITHUB_TOKEN` (*Add-ons → Secrets*, toggle **on** for this notebook) holding a GitHub token with *Contents: read & write* on your repo, and fill in the four values below. If they're left empty, this cell is skipped. A committed run ends by itself after this last cell, so no GPU time is wasted.
'''), code('''
GH_USER, GH_REPO, GH_BRANCH, GH_EMAIL = '', '', 'main', ''      # e.g. 'myname', 'my-repo', 'main', 'myname@users.noreply.github.com'
import shutil, subprocess, datetime
if not (GH_USER and GH_REPO):
    print('GitHub upload skipped (GH_USER / GH_REPO not set). Results are in the Output panel.')
else:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret('GITHUB_TOKEN')
    def run(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(token, '***')[-1500:]); return r
    # figures as PNG files
    from sosbench.viz import show_maps
    figdir = f'{RES}/figures'; os.makedirs(figdir, exist_ok=True)
    for n, (maps, gt) in MAPS.items():
        if maps:
            f = show_maps(maps, default_geometry(), gt=gt, ncol=9, size=2.0, title=n, vmin=1420, vmax=1640)
            f.savefig(f'{figdir}/compare_{n}.png', dpi=110, bbox_inches='tight'); plt.close(f)
    repo = '/kaggle/temp_repo'; shutil.rmtree(repo, ignore_errors=True)
    run(f'git clone --depth 1 -b {GH_BRANCH} https://{GH_USER}:{token}@github.com/{GH_USER}/{GH_REPO}.git {repo}')
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M'); dst = f'{repo}/sos_runs/{stamp}'
    for f in glob.glob(f'{RES}/**/*', recursive=True) + glob.glob(f'{RES2}/*') + glob.glob('log_*.txt') + glob.glob(f'{CKPT}/*.pt'):
        if os.path.isfile(f) and os.path.getsize(f) < 95e6:
            os.makedirs(os.path.join(dst, os.path.dirname(f)), exist_ok=True); shutil.copy(f, os.path.join(dst, f))
    if len(S): S.to_csv(f'{dst}/comparison_table.csv', index=False)
    run(f'git -C {repo} config user.name "{GH_USER}" && git -C {repo} config user.email "{GH_EMAIL or GH_USER + "@users.noreply.github.com"}"')
    run(f'git -C {repo} add -A && git -C {repo} commit -m "SoS comparison run {stamp}"')
    r = run(f'git -C {repo} push origin {GH_BRANCH}')
    print('PUSHED to', f'sos_runs/{stamp}' if r.returncode == 0 else 'FAILED - results remain in the Output panel')
print(f'total run time: {(time.time() - T_START) / 60:.0f} min')
''')]


def relabel(cell, old, new):
    c = dict(cell); c['source'] = c['source'].replace(old, new); return c


cells = (
    [INTRO, B.SETUP, CONFIG, B.DOWNLOAD, B.SMOKE]
    + THESIS_LAUNCH
    + [relabel(nb1[5], '## Step 1', '# Part A: DenOiS\\n## Step 1'.replace('\\n', '\n')), nb1[6], nb1[7], nb1[8]]
    + [DATAGEN_MD, nb1[10], nb1[11]]
    + [TRAIN_MD, TRAIN]
    + [relabel(nb1[14], '## Step 4', '## Part A, Step 4'), nb1[15], nb1[16], nb1[17], nb1[18], nb1[19], nb1[20]]
    + THESIS_WAIT + [nb2[7]]
    + [relabel(nb2[8], '## Head-to-head', '# Part C: head-to-head'), nb2[9], nb2[10], nb2[11], nb2[12], nb2[13]]
    + GITHUB
)

if __name__ == '__main__':
    out = os.path.join(B.HERE, 'notebooks', 'sos_denois_vs_thesis.ipynb')
    B.write(cells, out)

"""Plotting helpers for the notebooks."""
import numpy as np
import matplotlib.pyplot as plt

from .metrics import all_metrics


def show_maps(maps, geo, gt=None, vmin=1440, vmax=1620, ncol=6, title=None, bmode=None, size=2.6, metric_keys=('RMSE', 'ACE')):
    """maps: {name: (64,64) SoS}.  Annotates RMSE (ACE) when gt is given."""
    items = []
    if bmode is not None: items.append(('B-mode', bmode))
    if gt is not None and np.isfinite(gt).all(): items.append(('Ground truth', gt))
    items += list(maps.items())
    n = len(items); ncol = min(ncol, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * 1.15 * nrow), squeeze=False)
    ext = geo.extent_mm
    for ax, (name, img) in zip(axs.ravel(), items):
        if name == 'B-mode':
            ax.imshow(img, cmap='gray', vmin=-50, vmax=0, extent=ext, aspect='auto')
        else:
            im = ax.imshow(img, cmap='jet', vmin=vmin, vmax=vmax, extent=ext, aspect='auto')
            if gt is not None and np.isfinite(gt).all() and name != 'Ground truth':
                m = all_metrics(img, gt, geo)
                name = name + '\n' + ' / '.join(f'{k} {m[k]:.1f}' for k in metric_keys if k in m)
        ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax in axs.ravel()[n:]: ax.axis('off')
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label='SoS [m/s]')
    if title: fig.suptitle(title)
    return fig


def show_obs(obs, geo, w=None, pairs=(0, 3, 4, 7), vlim=30, title=None):
    """obs: {name: (8,64,64) delays [ns]}."""
    names = list(obs)
    fig, axs = plt.subplots(len(names), len(pairs), figsize=(2.4 * len(pairs), 2.4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, k in enumerate(pairs):
            img = obs[n][k].copy()
            if w is not None and i == 0: img = np.where(w[k] > 0, img, np.nan)
            im = axs[i, j].imshow(img, cmap='RdBu_r', vmin=-vlim, vmax=vlim, extent=geo.extent_mm, aspect='auto')
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j == 0: axs[i, j].set_ylabel(n, fontsize=8)
            if i == 0: axs[i, j].set_title(f'pair {k}', fontsize=8)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label='delay [ns]')
    if title: fig.suptitle(title)
    return fig


def table_frame(table, metrics=('RMSE', 'ACE', 'gCNR')):
    """{(input, recon): {metric: (mean, std)}} -> pandas DataFrame 'mean ± std'."""
    import pandas as pd
    rows = {}
    for (inp, rc), v in table.items():
        for m in metrics:
            if m in v:
                rows.setdefault((inp, m), {})[rc] = f'{v[m][0]:.2f} ± {v[m][1]:.2f}'
    return pd.DataFrame(rows).T

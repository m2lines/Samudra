"""Global raw-cell maps for the completed matched Adam local-range experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = Path('/tmp/om4-overfit-torch-input')
OUT = BASE / 'report/local-range'
mask = np.load(DATA / 'mask.npy').astype(bool)
target = np.load(DATA / 'prognostic.npy', mmap_mode='r')[2].astype(float)
arms = ['mse', 'spatial_100', 'local_range']
names = ['Ordinary MSE', 'Spatial-error loss', 'Local-range weighting']
preds = [np.load(BASE / f'results/local-range/{a}/maps-5000.npz')['prediction'][0].astype(float) for a in arms]
fields = [(57, 'Salinity, 2.5 m'), (70, 'Salinity, 1850 m'), (51, 'Temperature, 1850 m'), (13, 'Zonal velocity, 1850 m'), (76, 'SSH')]
summary = {}
fig, axs = plt.subplots(len(fields), 3, figsize=(15, 12), layout='constrained')
for row, (c, label) in enumerate(fields):
    errors = [p[c] - target[c] for p in preds]
    # Full extrema shared across arms: no clipped errors or per-panel autoscaling.
    limit = max(np.max(np.abs(e[mask[c]])) for e in errors)
    summary[label] = {}
    for col, (e, arm) in enumerate(zip(errors, arms)):
        mse = float(np.mean(e[mask[c]] ** 2))
        squared = np.sort(e[mask[c]] ** 2)
        top_count = max(1, int(np.ceil(.01 * len(squared))))
        summary[label][arm] = {'top_1_percent_share_squared_error': float(squared[-top_count:].sum() / squared.sum()), 'mse': mse, 'rmse': mse ** .5, 'max_abs_error': float(np.max(np.abs(e[mask[c]])))}
        im = axs[row, col].imshow(np.where(mask[c], e, np.nan), origin='lower', interpolation='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit)
        axs[row, col].set_title(f'{names[col]} · RMSE {mse ** .5:.4g}', fontsize=10)
        axs[row, col].set_facecolor('#b5b5b5')
        axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
    axs[row, 0].set_ylabel(label)
    fig.colorbar(im, ax=axs[row, :], shrink=.8, label='Error / training channel std')
fig.suptitle('Global prediction − target after 5,000 Adam updates\nShared symmetric full-extrema limits within each row; gray = land; raw grid cells, no smoothing')
fig.savefig(OUT / 'global-errors.png', dpi=160); plt.close(fig)
fig, axs = plt.subplots(len(fields), 3, figsize=(15, 12), layout='constrained')
for row, (c, label) in enumerate(fields):
    for col, p in enumerate(preds):
        im = axs[row, col].imshow(np.where(mask[c], p[c] - target[c], np.nan), origin='lower', interpolation='nearest', cmap='RdBu_r', vmin=-.1, vmax=.1)
        axs[row, col].set_facecolor('#b5b5b5'); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
        axs[row, col].set_title(names[col], fontsize=10)
    axs[row, 0].set_ylabel(label)
    fig.colorbar(im, ax=axs[row, :], shrink=.8, extend='both', label='Error / training channel std')
fig.suptitle('Global prediction − target · detail view\nSame ±0.1 training-std scale for every map; larger errors saturate; gray = land; no smoothing')
fig.savefig(OUT / 'global-errors-detail.png', dpi=160); plt.close(fig)
fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), layout='constrained')
for row, (c, label) in enumerate(fields[:2]):
    vals = [target[c], *[p[c] for p in preds]]
    lo = min(v[mask[c]].min() for v in vals); hi = max(v[mask[c]].max() for v in vals)
    for col, v in enumerate(vals):
        im = axs[row, col].imshow(np.where(mask[c], v, np.nan), origin='lower', interpolation='nearest', cmap='viridis', vmin=lo, vmax=hi)
        axs[row, col].set_facecolor('#b5b5b5'); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
        axs[row, col].set_title(['Target', *names][col], fontsize=10)
    axs[row, 0].set_ylabel(label)
    fig.colorbar(im, ax=axs[row, :], shrink=.7, label='Normalized salinity')
fig.suptitle('Global salinity fields · identical color limits within each row\nFull wet-cell extrema across target and all fits; raw grid cells, no smoothing')
fig.savefig(OUT / 'global-salinity.png', dpi=160); plt.close(fig)
(OUT / 'global-map-metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

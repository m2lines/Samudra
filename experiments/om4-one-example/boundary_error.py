"""Partition fixed state MSE by whether the spatial stencil is valid."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = {}
    contexts = {'seed15_oct': args.results / 'lbfgs-spatial'}
    contexts.update({name: args.results / 'spatial-confirm-lbfgs' / name for name in ('seed16_oct', 'seed15_jan')})
    for name, root in contexts.items():
        rows[name] = {}
        for arm in ('weight_0', 'weight_100'):
            with np.load(root / arm / 'maps-1000.npz') as m:
                wet = m['mask'].astype(bool)
                error = m['prediction'][0].astype(np.float64) - m['truth'][0].astype(np.float64)
            inside = wet.copy()
            for axis, shift in ((-2,-1),(-2,1),(-1,-1),(-1,1)):
                inside &= np.roll(wet, shift, axis)
            inside[:, (0,-1), :] = False
            edge = wet & ~inside
            denominator = wet.sum((-2,-1))
            total = np.mean((error**2 * wet).sum((-2,-1)) / denominator)
            components = {label: float(np.mean((error**2 * selection).sum((-2,-1)) / denominator)) for label, selection in [('interior', inside), ('stencil_invalid_wet_cells', edge)]}
            np.testing.assert_allclose(sum(components.values()), total, rtol=1e-12)
            rows[name][arm] = {'total_state_mse': float(total), 'contribution_to_state_mse': components,
                'stencil_invalid_error_fraction': components['stencil_invalid_wet_cells'] / total,
                'mean_per_channel_fraction_wet_stencil_invalid': float(np.mean(edge.sum((-2,-1)) / denominator))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'contexts': rows, 'definition': 'Exact additive partition of the existing equal-channel wet-cell MSE. Interior requires all four neighbors wet and excludes latitude endpoints; the other category is the remaining wet cells, not land. No causal attribution follows from this partition.'}, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, (name, arms) in zip(axes, rows.items()):
        interior = [arms[arm]['contribution_to_state_mse']['interior'] for arm in ('weight_0', 'weight_100')]
        edge = [arms[arm]['contribution_to_state_mse']['stencil_invalid_wet_cells'] for arm in ('weight_0', 'weight_100')]
        ax.bar([0, 1], interior, label='Interior')
        ax.bar([0, 1], edge, bottom=interior, label='Wet mask/latitude boundary')
        ax.set_xticks([0, 1], ['MSE only', 'Spatial loss'])
        ax.set_title(name.replace('_', ' '))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Contribution to equal-channel wet-cell state MSE')
    axes[-1].legend(fontsize=8)
    fig.suptitle('The interior-only spatial penalty redistributes residual error toward mask boundaries\n'
        'Matched L-BFGS; unchanged raw grids. Boundary category is 10.53% of wet cells, averaged over channels.')
    fig.savefig(args.output.with_suffix('.png'), dpi=170)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__=='__main__':
    main()

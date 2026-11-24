"""Audit coverage of the proposed wet-neighbor penalty against completed errors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lbfgs_coastal import wet_neighbor_curvature, interior_mask


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wet=torch.tensor(np.load(args.data/'mask.npy')[None]).bool()
    inside=interior_mask(wet)
    _,supported=wet_neighbor_curvature(torch.zeros(wet.shape),wet)
    counts=wet.sum((-2,-1)).double()
    extra=supported&~inside
    missing=wet&~supported
    coverage={
        'wet_cells_across_channels':int(wet.sum()),
        'interior_cells_across_channels':int(inside.sum()),
        'new_boundary_cells_across_channels':int(extra.sum()),
        'unsupported_wet_cells_across_channels':int(missing.sum()),
        'mean_per_channel_interior_fraction':float((inside.sum((-2,-1))/counts).mean()),
        'mean_per_channel_added_fraction':float((extra.sum((-2,-1))/counts).mean()),
        'mean_per_channel_unsupported_fraction':float((missing.sum((-2,-1))/counts).mean()),
        'unsupported_by_channel':missing.sum((-2,-1))[0].tolist(),
        'definition':'Original 77 depth-dependent masks. Extra term covers wet cells excluded by the original interior stencil that have at least one valid wet neighbor. Longitude wraps; latitude does not. Original state MSE supervises every wet cell including unsupported ones.'}
    (args.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    contexts={'seed15_oct':args.results/'lbfgs-spatial'}
    contexts.update({name:args.results/'spatial-confirm-lbfgs'/name for name in ('seed16_oct','seed15_jan')})
    categories={name:selection[0].numpy() for name,selection in [('interior',inside),('supported_boundary',extra),('isolated_wet',missing)]}
    denominator=counts[0].numpy()
    output={}
    for name,directory in contexts.items():
        output[name]={}
        for arm in ('weight_0','weight_100'):
            with np.load(directory/arm/'maps-1000.npz') as saved:
                assert np.array_equal(saved['mask'],wet[0].numpy())
                error=saved['prediction'][0].astype(np.float64)-saved['truth'][0].astype(np.float64)
            components={label:float(((error**2*selection).sum((-2,-1))/denominator).mean()) for label,selection in categories.items()}
            total=float(((error**2*wet[0].numpy()).sum((-2,-1))/denominator).mean())
            np.testing.assert_allclose(sum(components.values()),total,rtol=1e-12)
            output[name][arm]={'state_mse':total,'components':components,'isolated_fraction_of_total':components['isolated_wet']/total}
    (args.output/'error-coverage.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Verified coverage and exact additive error partition for all six endpoints')


if __name__=='__main__':
    main()

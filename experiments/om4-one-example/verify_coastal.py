"""Independently verify the boundary-loss objective and state-error partition."""
import argparse
import json
from pathlib import Path
import numpy as np


def components(prediction, truth, wet):
    error=prediction.astype(np.float64)-truth.astype(np.float64)
    inside=wet.copy()
    for axis,shift in ((-2,-1),(-2,1),(-1,-1),(-1,1)):
        inside &= np.roll(wet,shift,axis)
    inside[:,(0,-1),:]=False
    highpass=error-sum(np.roll(error,shift,axis) for axis,shift in ((-2,-1),(-2,1),(-1,-1),(-1,1)))/4
    total_wet=wet.sum((-2,-1))
    total_inside=inside.sum((-2,-1))
    # Compute a valid-neighbor average, independently of the Torch numerator implementation.
    neighbor_sum=np.zeros_like(error)
    neighbor_count=np.zeros_like(error,dtype=np.int32)
    for axis,shift in ((-2,-1),(-2,1),(-1,-1),(-1,1)):
        valid=wet & np.roll(wet,shift,axis)
        if axis==-2:
            valid[:, -1 if shift==-1 else 0, :]=False
        neighbor_sum += np.where(valid,np.roll(error,shift,axis),0)
        neighbor_count += valid
    supported=wet & (neighbor_count>0)
    boundary=supported & ~inside
    curvature=error-neighbor_sum/np.maximum(neighbor_count,1)
    state_per_channel=(error**2*wet).sum((-2,-1))/total_wet
    interior_per_channel=(error**2*inside).sum((-2,-1))/total_wet
    boundary_per_channel=(error**2*(wet&~inside)).sum((-2,-1))/total_wet
    np.testing.assert_allclose(interior_per_channel+boundary_per_channel,state_per_channel,rtol=1e-12)
    return {'state_mse':float(state_per_channel.mean()),
        'highpass_mse':float(((highpass**2*inside).sum((-2,-1))/total_inside).mean()),
        'extra_boundary_curvature_mse':float(((curvature**2*boundary).sum((-2,-1))/total_inside).mean()),
        'interior_state_mse_contribution':float(interior_per_channel.mean()),
        'boundary_state_mse_contribution':float(boundary_per_channel.mean()),
        'per_channel_interior_state_mse_contribution':interior_per_channel.tolist(),
        'per_channel_boundary_state_mse_contribution':boundary_per_channel.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-steps',type=int,default=1000)
    args=parser.parse_args()
    truth=np.load(args.data/'prognostic.npy',mmap_mode='r')[2]
    wet=np.load(args.data/'mask.npy').astype(bool)
    expected_source=None
    result={}
    for arm in ('interior','wet_neighbors'):
        path=args.root/arm
        assert (path/'COMPLETE').exists()
        protocol=json.loads((path/'protocol.json').read_text())
        assert protocol['boundary']==arm and protocol['weight']==100 and protocol['steps']==args.expected_steps
        if expected_source is None:expected_source=protocol['source']
        assert protocol['source']==expected_source
        with np.load(args.source/'maps-5000.npz') as source,np.load(path/'maps-0.npz') as initial:
            assert set(source.files)==set(initial.files)
            assert all(np.array_equal(source[k],initial[k]) for k in source.files)
        history=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
        end=history[-1]
        assert end['step']==args.expected_steps
        with np.load(path/f'maps-{args.expected_steps}.npz') as saved:
            assert np.array_equal(saved['truth'][0],truth) and np.array_equal(saved['mask'],wet)
            values=components(saved['prediction'][0],truth,wet)
        loss=values['state_mse']+100*values['highpass_mse']
        if arm=='wet_neighbors':loss+=100*values['extra_boundary_curvature_mse']
        np.testing.assert_allclose(loss,end['training_objective'],rtol=1e-5)
        for key in ('interior_state_mse_contribution','boundary_state_mse_contribution'):
            np.testing.assert_allclose(values[key],end[key],rtol=3e-6)
        values.update({'verified_training_objective':loss,'closure_evaluations':end['closure_evaluations'],
            'internal_iterations':end['internal_iterations'],'measured_seconds':end['seconds']})
        result[arm]=values
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'arms':result,'verification':'CPU float64 raw-map objective and additive wet-cell MSE partition; prepared truth/mask identity; exact initial-map identity with source endpoint. Run verify_maps.py separately for all77 state errors and target/increment ratios.'},indent=2)+'\n')
    print({arm:{key:v for key,v in values.items() if not key.startswith('per_channel')} for arm,values in result.items()})


if __name__=='__main__':main()

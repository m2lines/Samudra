"""Training-only smooth spatial/change scales for the local-range Adam test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d


def smooth(field, lat, dx, dy, radius_km):
    """Area-weighted separable geographic Gaussian; wet normalization is external.

    Latitude sigma is radius/111.195 degrees, zonal sigma adjusts for cos(latitude).
    Longitude wraps; latitude has no wrap. Truncate at 3 sigma. This is an
    approximate physical-distance kernel, not an exact spherical heat kernel.
    """
    tmp = gaussian_filter1d(field, radius_km/(111.195*dy), axis=0,
                           mode='constant', cval=0, truncate=3)
    out = np.empty_like(tmp)
    for j, phi in enumerate(lat):
        sigma = radius_km/(111.195*dx*max(np.cos(np.deg2rad(phi)),1e-3))
        out[j] = gaussian_filter1d(tmp[j], sigma, mode='wrap', truncate=3)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='/home/jder/data/om4_onedeg/OM4.zarr')
    ap.add_argument('--output',required=True)
    ap.add_argument('--samples',type=int,default=32)
    ap.add_argument('--radius-km',type=float,default=300)
    ap.add_argument('--floor',type=float,default=.02)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ds=xr.open_zarr(a.data,chunks=None)
    names=[f'{v}_{k}' for v in ['uo','vo','thetao','so'] for k in range(19)]+['zos']
    root=Path(a.data).parent
    means=xr.open_zarr(root/'OM4_means.zarr',chunks=None)
    stds=xr.open_zarr(root/'OM4_stds.zarr',chunks=None)
    meta={'means':[float(means[n].values) for n in names], 'stds':[float(stds[n].values) for n in names]}
    masks=np.stack([ds[f'mask_{i%19 if i<76 else 0}'].values.astype(bool) for i in range(77)])
    dates=ds.time.values
    eligible=[i for i in range(1,len(dates)) if '1975-01-03'<=str(dates[i-1])[:10] and str(dates[i])[:10]<'2013-09-30']
    selected=np.sort(np.random.default_rng(20261019).choice(eligible,a.samples,replace=False))
    lat,lon=np.asarray(ds.y),np.asarray(ds.x)
    dx=float(np.median(np.diff(lon)));dy=float(np.median(np.diff(lat)))
    area=np.cos(np.deg2rad(lat))[:,None]
    spatial=np.empty(masks.shape,dtype=np.float32);temporal=np.empty_like(spatial)
    variable_names=[f'{v}_{k}' for v in ['uo','vo','thetao','so'] for k in range(19)]+['zos']
    for c,name in enumerate(variable_names):
        wet=masks[c]
        depth=c%19 if c<76 else 0
        assert np.array_equal(wet,ds[f'mask_{depth}'].values),name
        den=smooth(wet*area,lat,dx,dy,a.radius_km)
        def avg(z):
            return smooth(np.where(wet,z,0)*area,lat,dx,dy,a.radius_km)/np.maximum(den,1e-30)
        sp=np.zeros(wet.shape);tp=np.zeros_like(sp)
        for i in selected:
            assert (dates[i]-dates[i-1]).days==5
            pair=ds[name].isel(time=[int(i-1),int(i)]).values.astype(np.float64)
            assert np.isfinite(pair[:,wet]).all(),(name,int(i))
            normalized=(pair-meta['means'][c])/meta['stds'][c]
            z=normalized[1];delta=normalized[1]-normalized[0]
            mean=avg(z)
            sp+=np.maximum(avg(z*z)-mean*mean,0)
            tp+=avg(delta*delta)
        spatial[c]=np.where(wet,sp/a.samples,0)
        temporal[c]=np.where(wet,tp/a.samples,0)
        if c%10==0:print(f'Prepared {c+1}/77: {name}',flush=True)
    scale2=np.maximum(np.maximum(spatial,temporal),a.floor**2)
    weights=np.where(masks,1/scale2,0)
    average=float((weights.sum((-2,-1))/masks.sum((-2,-1))).mean())
    weights/=average
    from bounded import bounded_weights
    bounded=bounded_weights(np.maximum(spatial,temporal),masks,.5,2.)
    weights=np.where(masks,1+.1*(bounded-1),0).astype(np.float32)
    assert np.isfinite(weights).all() and (weights[masks]>0).all()
    assert abs(float((weights.sum((-2,-1))/masks.sum((-2,-1))).mean())-1)<1e-6
    np.savez_compressed(out/'scales.npz',weights=weights,spatial_variance=spatial,
                        temporal_change_mse=temporal,mask=masks,scale2=scale2)
    record=vars(a)|{'source_indices':selected.tolist(),
        'date_pairs':[[str(dates[i-1]),str(dates[i])] for i in selected],
        'training_bounds':['1975-01-03','2013-09-30 exclusive'],
        'normalization_means':meta['means'][:77],'normalization_stds':meta['stds'][:77],
        'raw_average_weight_before_bounded_transform':average,
        'channel_names':names,
        'weighted_mean_per_channel':(weights.sum((-2,-1))/masks.sum((-2,-1))).tolist(),
        'floor_fraction_per_channel':(((spatial<a.floor**2)&(temporal<a.floor**2)&masks).sum((-2,-1))/masks.sum((-2,-1))).tolist(),
        'global_weight_percentiles':np.percentile(weights[masks],[0,1,50,99,100]).tolist(),
        'kernel':'separable Gaussian sigma300km; cos(latitude) zonal correction, area/wet normalization; no land observations but kernel may span narrow land barriers',
        'objective':'current-main masked full-grid MSE with fixed mild10 weights; per-channel wet mean1, bounds0.95-1.10',
        'scale_file_sha256':hashlib.sha256((out/'scales.npz').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Saved',out,flush=True)

if __name__=='__main__':main()

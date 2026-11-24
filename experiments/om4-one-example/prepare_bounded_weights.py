"""Bound local priorities while preserving equal mean weight for every channel."""
import json
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent

def bounded_weights(variance, mask, power, cap):
    raw = np.maximum(variance, .02 ** 2).astype(np.float64) ** (-power)
    out = np.zeros_like(raw)
    for c in range(len(raw)):
        values = raw[c][mask[c]]
        lo, hi = 0., cap / values.min()
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.clip(values * mid, 1 / cap, cap).mean() > 1:
                hi = mid
            else:
                lo = mid
        out[c][mask[c]] = np.clip(values * ((lo + hi) / 2), 1 / cap, cap)
    return out.astype(np.float32)

if __name__ == '__main__':
    source = np.load(BASE / 'results/local-range-scales/scales.npz')
    mask = source['mask'].astype(bool)
    variance = np.maximum(source['spatial_variance'], source['temporal_change_mse'])
    dest = BASE / 'results/bounded-range-scales'; dest.mkdir(exist_ok=True)
    protocol = {}
    for name, power, cap in [('sqrt_cap2', .5, 2), ('sqrt_cap4', .5, 4), ('inverse_cap4', 1., 4)]:
        w = bounded_weights(variance, mask, power, cap)
        means = w.sum((-2, -1)) / mask.sum((-2, -1))
        np.testing.assert_allclose(means, 1, rtol=1e-6)
        assert w[mask].min() >= 1/cap and w[mask].max() <= cap
        assert (w[~mask] == 0).all()
        np.savez_compressed(dest / f'{name}.npz', weights=w, mask=mask)
        protocol[name] = {'power_on_variance':power, 'min_weight':float(w[mask].min()), 'max_weight':float(w[mask].max()), 'per_channel_mean':means.tolist(), 'floor_std':.02, 'cap':cap}
    (dest / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('Passed per-channel mean, bounds, land-mask checks for three weight sets.')

"""Bound local priorities while preserving equal mean weight for every channel."""
import numpy as np

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


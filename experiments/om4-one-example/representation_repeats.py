"""Repeat the representation controls without changing their training loop."""

import representation_sweep

# Complete the missing latest-input/random-head cell of the seed-15 comparison.
# Seed-16/17 runs reuse the original absolute/random and residual/zero arms.
representation_sweep.ARMS.append(("latest_random", 77, False))

if __name__ == "__main__":
    representation_sweep.main()
